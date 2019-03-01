"""Tools for detecting collision templates."""

import copy as copy
import numpy as np
from tqdm import tqdm

from yass.deconvolve.optimized_match_pursuit import OptimizedMatchPursuit
from yass.deconvolve.template import WaveForms


def MAD_bernoulli_two_uniforms(a, b):
    """Analytical MAD of a addition of uniform drawn according to Bernoulli rv.

    i ~ Bernoulli(0.5)
    x ~ uniform(0., a)
    y ~ uniform(0., b)
    z = ix + (1-i)y
    This functions computes the analytical median absolute deviation
    of the resulting z random variable.

    params:
    -------
    a: int or np.ndarray
    b: int or np.ndarray

    returns:
    --------
    float or np.ndarray
    """
    same_sign_indicator = np.array(a * b > 0.).astype(np.float)
    a, b = np.abs(a), np.abs(b)
    # Make sure that a , b on all elements of the arrays
    concat = np.array([a, b])
    a, b = concat.min(axis=0), concat.max(axis=0)
    # Median of the random variable
    a_inv, b_inv = 1. / a, 1. / b
    a_inv_plus_b_inv = a_inv + b_inv
    m = 1. / a_inv_plus_b_inv
    t1 = (a - m) * a_inv_plus_b_inv
    t2 = (2 * m - a) * (a_inv / 2. + b_inv)
    # remainder is t3 = (b - 2*m) * 1/b / 2. where t1 + t2 + t3 = 1

    # handle two where t1 less than or greater than 0.5
    indicator = np.array(t1 > 0.5).astype(np.float)
    opt1 = (.5 / t1) * (a - m)
    opt2 = (a - m) + (0.5 - t1) / t2 * (2 * m - a)
    out = indicator * opt1 + (1 - indicator) * opt2
    # handle two cases where a and b have different signs
    return same_sign_indicator * out + (1 - same_sign_indicator) * m


def collision_templates(
        templates, get_unit_spike_fun, mad_gap=0.8, mad_gap_breach=4,
        residual_max_norm=1.):
    """Given templates and spikes determines collision templates.

    params:
    -------
    templates: np.ndarray
        Has shape (# units, # channels, # number of samples).
    get_unit_spike_fun: function
        Function that as parameter gets a unit id which is int, and returns
        spikes that have shape (#spikes, #channels, # time samples).
    mad_gap: float
        The threshold between analytical MAD of templates  and MAD statistics
        of spikes.
    mad_gap_breach: int
        Total number of allowable time points-channels that must not satisfy
        the gap between analytical MAD and computed MAD so that a template is
        classified as collision.
    residual_max_norm: float
        Threshold for residual of reconstructed template using deconvolution
        below which the templates is classified as collision.

    returns:
    --------
    np.array of int. List of unit ids that are determined to be
    collisions using deconvolution. Second, and unit ids that
    have unexpected MAD values.
    """
    n_unit, n_chan, n_time = templates.shape

    # First step is to deconvolve given templates using the rest of them.
    # This steps gets rid of templates that are reconstructed very well using
    # the rest.

    # Stack all the templates temporally given enough space between them.
    data = np.zeros([n_time * 2 * n_unit, n_chan])
    tot_time = data.shape[0]
    spike_times = np.arange(n_time // 2, tot_time, n_time * 2) + 10
    for unit, time in enumerate(spike_times):
        data[time:time + n_time, :] += templates[unit].T
    mp = OptimizedMatchPursuit(
            data, templates.transpose([2, 1, 0]), threshold=0,
            conv_approx_rank=10, vis_su=1., upsample=16, keep_iterations=True)
    mp.compute_objective()
    # Before the subtraction step of match-pusuit guarantee that no template
    # is deconvolved with itself.
    for unit, time in enumerate(spike_times):
        time = time + n_time // 2
        mp.obj[unit, time:time + n_time] = - np.inf

    deconvd_sp, dist_metric = mp.run(max_iter=2)
    otemps = mp.get_upsampled_templates()
    ospt = mp.correct_shift_deconv_spike_train()


    residual = copy.copy(data)
    for up_unit in np.unique(ospt[:, 1]):
        # Get spike times
        unit_spt = ospt[ospt[:, 1] == up_unit, :1]
        residual[np.arange(n_time) + unit_spt, :] -= otemps[:, :, up_unit]

    unit_residuals = []
    unit_factors = []
    for unit in tqdm(range(n_unit), "Computing Residuals"):
        time = spike_times[unit]
        factor_times = np.logical_and(
                ospt[:, 0] > time - n_time, ospt[:, 0] < time + n_time)
        factor = ospt[factor_times, 1]
        # Convert upsample id to original template/unit id.
        factor = factor // 16
        unit_factors.append(factor)
        window = 10
        spike_res = residual[time+window:time+n_time-window] 
        unit_residuals.append(spike_res)

    unit_residuals = np.array(unit_residuals)
    res_max_norm = np.abs(unit_residuals).max(axis=-1).max(axis=-1)
    # Get those templates that have residual max norm of no more that 1.
    # as collision templates.
    candidates = np.where(res_max_norm < residual_max_norm)[0]

    # Keep track of elemental units necessary to reconstruct collision
    # templates.
    elemental = []
    deconv_collision_picks = []
    for unit in candidates:
        if unit not in elemental:
            deconv_collision_picks.append(unit)
            for e in unit_factors[unit]:
                if e not in elemental:
                    elemental.append(e)

    # Second Stage: use MAD statistics of unit wave forms to detect
    # Collision templates.
    unit_mads = []
    # Analytical AMD if noise of spike is only uniform jitters. This
    # serves as a lower bound for the actual MAD.
    unit_emads = []

    # Jitter for alignments for spike wave forms.
    jitter = 3
    visch = (templates.ptp(axis=-1) > 2.)
    for unit in tqdm(range(n_unit), "Computing MAD"):
        wf = get_unit_spike_fun(unit)
        wf = wf[:, visch[unit], :]
        mean = wf.mean(axis=0)
        # Compute the approximate expected MAD of normali looking spikes.
        wf = WaveForms(wf).align(jitter=jitter)
        # Clipp the mean so that it is the same shape as the aligned
        # waveforms.
        mean_clipped = mean[:, jitter:-jitter]
        delta_minus = np.abs(mean_clipped - mean[:, jitter-1:-jitter-1])
        delta_plus = np.abs(mean_clipped - mean[:, jitter+1:-jitter+1])

        emad = MAD_bernoulli_two_uniforms(delta_minus, delta_plus)
        unit_emads.append(emad)

        t_mad = np.median(
                np.abs(np.median(wf, axis=0)[None, :, :] - wf), axis=0)
        unit_mads.append(t_mad)

    # Count total number of unexpectedly large MAD values per unit.
    high_mad_counts = []
    for unit in range(n_unit):
        high_mad_counts.append(
                ((unit_mads[unit] - unit_emads[unit]) > mad_gap).sum())
    high_mad_counts = np.array(high_mad_counts)
    mad_collision_picks = np.where(high_mad_counts > mad_gap_breach)[0]
    # remove elemental units
    mad_collision_picks = np.setdiff1d(
            mad_collision_picks, np.array(elemental))

    return np.union1d(
            np.array(deconv_collision_picks),
            np.array(mad_collision_picks))

