from yass.cluster import align_get_shifts_with_ref, shift_chans

def plot_normalized_templates(templates, neigh_channels, ref_template, shift_allowance):

	"""
	plot normalized templates on their main channels and secondary channels
	templates: number of channels x temporal window x number of units
	geometry: number of channels x 2
	"""

	C, R, K = templates.shape
	mc = np.argmmax(templates.ptp(1), 0)

	# get main channel templates
	templates_mc = np.zeros((K, R))
	for k in range(K):
		templates_mc[k] = templates[mc[k], :, k]

	# shift templates_mc
	best_shifts_mc = align_get_shifts_with_ref(
	                templates_mc,
	                ref_template)
    templates_mc = shift_chans(templates_mc, best_shifts_mc)

    # normalize templates
    norm_mc = np.linalg.norm(templates_mc, axis=1, keepdims=True)
    templates_mc /= norm_mc

    # get secdonary channel templates
	templates_sec = np.zeros((0, R))
	for k in range(K):
	    neighs = np.copy(neigh_channels[mc[k]])
	    neighs[mc[j]] = False
	    neighs = np.where(neighs)[0]
	    templates_sec = np.concatenate((templates_sec, templates[k, :, neighs]), axis=0)

	# shift templates_sec
	best_shifts_sec = align_get_shifts_with_ref(
	                templates_sec,
	                ref_template)
    templates_sec = shift_chans(templates_sec, best_shifts_sec)

    # normalize templates
    norm_sec = np.linalg.norm(templates_sec, axis=1, keepdims=True)
    templates_sec /= norm_sec

    plt.figure(figsize=(5,3))
	plt.plot(templates_mc.T, color='k', alpha=0.1)
	plt.title('aligned normalized templates on their max channel')
	plt.savefig('aligned_norm_mc_templates.png')


	ths = [0,1,3,5]
	plt.figure(figsize=(20,5))
	for ii, th in enumerate(ths):
	    plt.subplot(1,4,ii+1)
	    idx = templates_sec.ptp(1) > th
	    plt.plot(aligned_templates_sec_norm[idx].T, color='k', alpha=0.1)
	    plt.xlabel('templates with ptp > '+str(th), fontsize=15)

	plt.suptitle('aligned templates on neighbors of their max channel', fontsize=30)
	plt.savefig('aligned_norm_neigh_templates.png')
	





