import numpy as np
from scipy.stats import gamma,poisson
from gnmf import nmf_vb
import matplotlib.pyplot as plt

if __name__ == '__main__':
	W=40
	K=30
	I=3
	a_tm=np.ones((W,I))*10
	b_tm=np.ones((W,I))
	a_ve=np.ones((I,K))*0.1
	b_ve=np.ones((I,K))
	T = gamma.rvs(a_tm,b_tm)
	V = gamma.rvs(a_ve,b_ve)
	X = poisson.rvs(T@V)
	X = X + 1000*np.random.randn(*X.shape)
	(tie_a_tm, tie_b_tm, tie_a_ve, tie_b_ve)=('free','tie_all','free','tie_all')
	init_mat_factor = nmf_vb(a_tm, b_tm, a_ve, b_ve,tie_a_tm, tie_b_tm, tie_a_ve, tie_b_ve)
	g=init_mat_factor.VB_parm_calc(X,epoch=1000,update=10,print_period=500)
	(g_E_T,g_E_logT,g_E_V,g_E_logV,g_Bound,g_a_ve,g_b_ve,g_a_tm,g_b_tm)=g
	X_reconstruct=g_E_T@g_E_V
	#upper left plot is the reconstructed matrix
	plt.subplot(211)
	plt.imshow(X_reconstruct)
	#lower left plot is the original matrix
	plt.subplot(212)
	plt.imshow(X)
	#the right subplot is a colorbar
	plt.subplots_adjust(bottom=0.1,right=0.8,top=0.9)
	cax=plt.axes([0.6,0.1,0.075,0.8])
	plt.colorbar(cax=cax)
	plt.show()