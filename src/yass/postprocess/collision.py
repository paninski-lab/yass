import numpy as np
import os
import parmap

from yass.postprocess.util import run_deconv

def remove_collision(fname_templates, save_dir, units_in=None,
                      multi_processing=False, n_processors=1):

    # output folder
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # units_in is all units if none
    if units_in is None:
        n_units = np.load(fname_templates).shape[0]
        units_in = np.arange(units_in)

    # collect save file name
    fnames_out = []
    for unit in units_in:
        fname = os.path.join(save_dir, 'unit_{}.npz'.format(unit))
        fnames_out.append(fname)

    # run deconv on template
    if multi_processing:
        parmap.starmap(deconv_on_template,
                       list(zip(units_in, fnames_out)),
                       units_in,
                       fname_templates,
                       processes=n_processors,
                       pm_pbar=True)      
    else:
        for ctr in range(len(units_in)):
            deconv_on_template(units_in[ctr],
                               fnames_out[ctr],
                               units_in, 
                               fname_templates)

    # gather result
    units_kill = np.zeros(len(units_in), 'bool')
    for ctr in range(len(units_in)):

        if np.load(fnames_out[ctr])['collision']:
            units_kill[ctr] = True

    # 
    units_keep = units_in[~units_kill]
    return units_keep


def deconv_on_template(unit, fname_out, units_in, fname_templates,
                       up_factor=8, residual_max_norm=1.2):

    if os.path.exists(fname_out):
        return

    # load templates
    templates = np.load(fname_templates)
    n_units = templates.shape[0]

    # unit to be tested
    data = templates[unit]

    # templates to run on
    units_ = units_in[units_in != unit]
    templates = templates[units_]

    # data noise chan
    noise_chan = np.max(
        np.abs(data), axis=0) < residual_max_norm/2

    # if no noise chans, select max chan
    if noise_chan.sum()==0:
        idx_include = np.ones(templates.shape[0], 'bool')
    else:
        # exclude units that won't help deconvolve this unit
        # if it has large energy on noise chan, then it should not help
        #print (" n_units: ", n_units, 'templates: ', templates.shape, 'noise: ', 
        #        noise_chan.sum())
        idx_include = np.abs(templates[:, :, noise_chan]
                            ).max(axis=(1, 2)) < residual_max_norm/2
    templates = templates[idx_include]
    units_ = units_[idx_include]

    if templates.shape[0] == 0:
        # save result
        np.savez(fname_out,
                 collision=False,
                 residual=None,
                 deconv_units=None)
        return

    # visible channels only
    vis_chan = np.where(~noise_chan)[0]
    data = data[:, vis_chan]
    templates = templates[:, :, vis_chan]

    # run at most 3 iterations 
    it, max_it = 0, 3

    # run deconv
    collision = False
    deconv_units = []
    while it < max_it and not collision:

        # run one iteration of deconv
        residual, best_fit_unit = run_deconv(data, templates, up_factor, 'l2')

        # if nothing fits more, quit
        if best_fit_unit is None:
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
    
    # save result
    np.savez(fname_out,
             collision=collision,
             residual=residual,
             deconv_units=deconv_units)
