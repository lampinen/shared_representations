import numpy as np
import matplotlib.pyplot as plot
from scipy.spatial.distance import pdist, squareform

nruns = 5
ndomains = 2

for rseed in xrange(nruns):
    figure_num = 0
    for nlayer in [3,2]:
	for network in ['nonlinear']:#, 'linear']:
    #        for ndomains in [1,2,3]:
	    print network, rseed 
	    plot.figure(figure_num)
	    filename_prefix = "results/depth_and_ndom_comp/%s_nlayer_%i_ndomains_%i_rseed_%i_" %(network,nlayer,ndomains,rseed)
	    X = np.genfromtxt(filename_prefix + 'epoch_9900_internal_rep.csv', delimiter=',')
	    simils = squareform(pdist(X))
	    plot.imshow(simils, interpolation='none', vmin=0, vmax=2)
	    plot.colorbar()
	    plot.title(network + (' nlayer = %i' % nlayer) + ' internal rep')
	    figure_num += 1

	    plot.figure(figure_num)
	    X = np.genfromtxt(filename_prefix + 'epoch_9900_pre_outputs.csv', delimiter=',')
	    simils = squareform(pdist(X))
	    plot.imshow(simils, interpolation='none', vmin=0, vmax=4, cmap='Greys')
	    plot.colorbar()
	    plot.title(network + (' nlayer = %i' % nlayer) + ' pre-output')
	    figure_num += 1
    plot.show()
	

