import numpy as np
import os
import parmap

from yass.postprocess.util import run_deconv

def remove_collision(fname_templates, save_dir, CONFIG, units_in=None,
                      multi_processing=False, n_processors=1):

    # output folder
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # units_in is all units if none
    if units_in is None:
        n_units = np.load(fname_templates).shape[0]
        units_in = np.arange(units_in)
    units_in = np.array(units_in)

    up_factor = 8
    residual_max_norm = CONFIG.clean_up.abs_max_diff

    fname_units_kill = os.path.join(save_dir, 'units_kill.npy')
    if os.path.exists(fname_units_kill):
        units_kill = np.load(fname_units_kill)

    else:
        # run deconv on template
        if multi_processing:
            # split units_in
            units_in_split = []
            for j in range(n_processors):
                n_units_in = len(units_in)
                units_in_split.append(
                    np.array(units_in[slice(j, n_units_in, n_processors)]))

            units_kill = parmap.map(
                deconv_on_template,
                units_in_split,
                units_in,
                fname_templates,
                up_factor,
                residual_max_norm,
                processes=n_processors)
            units_kill = np.hstack(units_kill)

        else:
            units_kill = deconv_on_template(
                units_in,
                units_in,
                fname_templates,
                up_factor,
                residual_max_norm)
        np.save(fname_units_kill, units_kill)

    return np.setdiff1d(units_in, units_kill)


def deconv_on_template(units_test, units_in, fname_templates,
                       up_factor=8, residual_max_norm=1.2):


    # load templates
    templates = np.load(fname_templates)

    units_kill = []
    for unit in units_test:

        # unit to be tested
        data = templates[unit]

        # templates to run on
        units_ = units_in[units_in != unit]
        templates_ = templates[units_]

        # data noise chan
        noise_chan = np.max(
            np.abs(data), axis=0) < residual_max_norm/2

        # if no noise chans, select max chan
        if noise_chan.sum() > 0:
            # exclude units that won't help deconvolve this unit
            # if it has large energy on noise chan, then it should not help
            #print (" n_units: ", n_units, 'templates: ', templates.shape, 'noise: ',
            #        noise_chan.sum())
            idx_include = np.abs(templates_[:, :, noise_chan]
                                ).max(axis=(1, 2)) < residual_max_norm/2
            templates_ = templates_[idx_include]
            units_ = units_[idx_include]

        # no templates to deconvolve
        if templates_.shape[0] == 0:
            continue

        # visible channels only
        vis_chan = np.where(~noise_chan)[0]
        data = data[:, vis_chan]
        templates_ = templates_[:, :, vis_chan]

        # run at most 3 iterations
        it, max_it = 0, 3

        # run deconv
        collision = False
        deconv_units = []
        while it < max_it and not collision:

            # run one iteration of deconv
            residual, best_fit_unit, obj = run_deconv(data, templates_, up_factor, 'l2')

            # if nothing fits more, quit
            if obj < 0:
                it = max_it

            # if residual is small enough, stop
            elif np.max(np.abs(residual)) < residual_max_norm:
                deconv_units.append(units_[best_fit_unit])
                collision = True

            # if there is a fit but residual is still large,
            # run another iteration on residual
            else:
                data = residual
                deconv_units.append(units_[best_fit_unit])
                it += 1

        if collision:
            units_kill.append(unit)

    return np.array(units_kill)
