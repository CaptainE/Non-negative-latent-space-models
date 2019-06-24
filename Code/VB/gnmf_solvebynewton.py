from scipy import log
from scipy.special import polygamma
import numpy as np

def gnmf_solvebynewton(c,a0):
	"""
	routine to solve C=Log(A)-Psi(A)+1 function by newtons method
	"""
	M,N = a0.shape
	Mc,Nc = c.shape

	if M == Mc and N == Nc:
		a = a0
		cond = 1
	else:
		a = a0[0,0]
		cond = 4
	stop = False
	for i in range(10):
		delta = (log(a) - polygamma(0,a) + 1 - c) / ((1 / a) - polygamma(1,a))
		#print(delta.shape)
		count = 0
		while (delta > a).any():
			delta = delta / 2
			if count > 10:
				stop = True
				break
			count += 1
		if stop:
			break
		if (delta < 0).any():
			delta = 0
		a = a - delta
	if cond == 4:
		a = a * np.ones((M,N))

	return a