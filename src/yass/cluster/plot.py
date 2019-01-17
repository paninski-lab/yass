from yass.cluster.cluster import align_get_shifts_with_ref, shift_chans
from yass.util import absolute_path_to_asset

import os
import numpy as np
import matplotlib.pyplot as plt
matplotlib.use('Agg')


def plot_normalized_templates(templates_dir, neigh_channels):

	"""
	plot normalized templates on their main channels and secondary channels
	templates: number of channels x temporal window x number of units
	geometry: number of channels x 2
	"""

	ref_template = np.load(absolute_path_to_asset(os.path.join('template_space', 'ref_template.npy')))


	templates_pre_merge = np.load(templates_dir+'/templates_post_cluster_pre_merge.npy')
	templates_post_merge = np.load(templates_dir+'/templates_post_cluster_post_merge.npy')
	collisions = np.load(templates_dir+'/collisions.npy')
	templates_col_removed = templates_post_merge[~collisions]

	shift = (templates_pre_merge.shape[1] - len(ref_template))//2

	templates_pre_merge = templates_pre_merge[:, shift:-shift]
	templates_post_merge = templates_post_merge[:, shift:-shift]
	templates_col_removed = templates_col_removed[:, shift:-shift]

	templates_main = [None]*3
	templates_sec = [None]*3
	ptp_main = [None]*3
	ptp_sec = [None]*3

	templates_main[0], templates_sec[0], ptp_main[0], ptp_sec[0]  = get_normalized_templates(templates_pre_merge, neigh_channels, ref_template)
	templates_main[1], templates_sec[1], ptp_main[1], ptp_sec[1] = get_normalized_templates(templates_post_merge, neigh_channels, ref_template)
	templates_main[2], templates_sec[2], ptp_main[2], ptp_sec[2] = get_normalized_templates(templates_col_removed, neigh_channels, ref_template)

	templates_label = ['pre-merge', 'post-merge', 'post collision removal']

	plt.figure(figsize=(17, 7))
	for j in range(3):
		plt.subplot(1,3,j+1)
		plt.plot(templates_main[j].T, color='k', alpha=0.1)
		plt.xlabel(templates_label[j])
	plt.suptitle('aligned normalized templates on their max channel')
	plt.savefig(templates_dir+'/figures/aligned_norm_mc_templates.png')


	ths = [0,1,3]
	count = 1
	plt.figure(figsize=(17, 17))
	for j in range(3):
		for ii, th in enumerate(ths):
		    plt.subplot(3, 3, count)
		    idx = ptp_sec[j] > th
		    plt.plot(templates_sec[j][idx].T, color='k', alpha=0.1)
		    if j==2:
		    	plt.xlabel('ptp > '+str(th), fontsize=15)
		    if ii == 0:
		    	plt.ylabel(templates_label[j])
		    count += 1

	plt.suptitle('aligned templates on neighbors of their max channel', fontsize=30)
	plt.savefig(templates_dir+'/figures/aligned_norm_neigh_templates.png')


def get_normalized_templates(templates, neigh_channels, ref_template):

	"""
	plot normalized templates on their main channels and secondary channels
	templates: number of channels x temporal window x number of units
	geometry: number of channels x 2
	"""

	K, R, C = templates.shape
	mc = np.argmax(templates.ptp(1), 1)

	# get main channel templates
	templates_mc = np.zeros((K, R))
	for k in range(K):
		templates_mc[k] = templates[k, :, mc[k]]

	# shift templates_mc
	best_shifts_mc = align_get_shifts_with_ref(
	                templates_mc,
	                ref_template)
	templates_mc = shift_chans(templates_mc, best_shifts_mc)
	ptp_mc = templates_mc.ptp(1)

	# normalize templates
	norm_mc = np.linalg.norm(templates_mc, axis=1, keepdims=True)
	templates_mc /= norm_mc

	# get secdonary channel templates
	templates_sec = np.zeros((0, R))
	for k in range(K):
	    neighs = np.copy(neigh_channels[mc[k]])
	    neighs[mc[k]] = False
	    neighs = np.where(neighs)[0]
	    templates_sec = np.concatenate((templates_sec, templates[k, :, neighs]), axis=0)

	# shift templates_sec
	best_shifts_sec = align_get_shifts_with_ref(
	                templates_sec,
	                ref_template)
	templates_sec = shift_chans(templates_sec, best_shifts_sec)
	ptp_sec = templates_sec.ptp(1)

	# normalize templates
	norm_sec = np.linalg.norm(templates_sec, axis=1, keepdims=True)
	templates_sec /= norm_sec

	return templates_mc, templates_sec, ptp_mc, ptp_sec