import numpy as np
from scipy.stats import gamma,poisson
from gnmf import nmf_vb
import matplotlib.pyplot as plt
from scipy.io import loadmat

if __name__ == '__main__':

	"""
	Load generated data
	"""
	data = loadmat('../../Data/gendata.mat')
	X = data['X']
	X_true = data['DD']

	"""
	Model set-up
	"""
	I = 2
	W = X.shape[0]
	K = X.shape[1]

	a_tm = np.ones((W,I))*10
	b_tm = np.ones((W,I))
	a_ve = np.ones((I,K))*0.05
	b_ve = np.ones((I,K))

	(tie_a_tm, tie_b_tm, tie_a_ve, tie_b_ve) = ('free','clamp','free','clamp')#('clamp','clamp','clamp','clamp')
	model = nmf_vb(a_tm, b_tm, a_ve, b_ve, tie_a_tm, tie_b_tm, tie_a_ve, tie_b_ve)

	g = model.VB_parm_calc(X,epoch=1000,update=100,print_period=200)

	(D,g_E_logT,H,g_E_logV,g_Bound,g_a_ve,g_b_ve,g_a_tm,g_b_tm) = g

	X_reconstruct = D @ H

	plot = True
	#upper left plot is the reconstructed matrix
	if plot:
		#plt.subplot(131)
		plt.figure()
		plt.imshow(X_reconstruct,aspect='auto')
		plt.title("Recon")
		plt.axis("off")
		plt.savefig('vb_X.eps',bbox_inches="tight")

		#lower left plot is the original matrix
		#plt.subplot(132)
		plt.figure()
		plt.imshow(X,aspect='auto')
		plt.title("Noise")
		plt.axis("off")

		#plt.subplot(133)
		plt.figure()
		plt.imshow(X_true,aspect='auto')
		plt.title("True")
		plt.axis("off")

		plt.figure()
		plt.title('VB')
		plt.plot(D[:,0])

		plt.figure()
		plt.title('VB')
		plt.plot(H.T[:,0])
		plt.savefig('vb_spectrum.eps',bbox_inches="tight")


		plt.show()
