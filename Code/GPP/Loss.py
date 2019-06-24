# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 15:21:43 2018

@author: Niklas
"""
import utils
import numpy as np
from scipy.optimize import minimize
import covariance_functions as cvf

class Loss:
	def __init__(self,X,M,d_link,h_link,sigN,kernel=cvf.rbf,cov_pars=(0.001,0.01)):
		self.X = X
		self.K, self.L = X.shape
		self.M = M
		self.d_link = d_link
		self.h_link = h_link
		self.sigN = sigN
		self.kernel = kernel
		self.d_cov_par = cov_pars[0]
		self.h_cov_par = cov_pars[1]

		d_cov,h_cov = self._getCovarianceMatrices()

		self.Cd = utils.cholesky(d_cov)
		self.Ch = utils.cholesky(h_cov)

	def _getCovarianceMatrices(self):
		"""
		Computes the Covariance Matrices for the given kernel
		"""
		KM = self.K*self.M
		ML = self.M*self.L
		# Initialize
		#d_cov = np.zeros( (KM , KM) )
		#h_cov = np.zeros( (ML , ML) )
		# Noise to prevent diagonal of zeros
		eps = 1e-10
		# Generate covariance matrix of d
		m1 = np.arange(1,KM+1).reshape(KM,1)
		M1 = np.repeat(m1,KM,axis=1)
		M2 = M1.T
		d_cov = self.kernel(M1,M2,self.d_cov_par) + eps*np.eye(KM)
		print("Done generating Cov_D")
		#for i in range(KM):
		#	for j in range(KM):
		#		d_cov[i,j] = self.kernel(i+1,j+1,self.d_cov_par)
		#d_cov += eps*np.eye(KM)
		# Generate covariance matrix of h
		n1 = np.arange(1,ML+1).reshape(ML,1)
		N1 = np.repeat(n1,ML,axis=1)
		N2 = N1.T
		h_cov = self.kernel(N1,N2,self.h_cov_par) + eps*np.eye(ML)
		print("Done generating Cov_H")
		#for i in range(ML):
		#	for j in range(ML):
		#		h_cov[i,j] = self.kernel(i+1,j+1,self.h_cov_par)
		#h_cov += eps*np.eye(ML)
		return d_cov,h_cov

	def _vars_to_factors(self,delta,eta):
		D = utils.vec_inv( self.d_link.inverse( self.Cd.T @ delta ), (self.K,self.M) )
		H = utils.vec_inv( self.h_link.inverse( self.Ch.T @ eta ) , (self.M,self.L) )

		return D,H

	def _likelihood(self,D,H):
		"""
		Computes the least squares likelihood
		"""
		return 0.5 * ( self.sigN**(-2) *  np.linalg.norm( self.X - D @ H)**2)

	def total_loss(self,delta,eta,reverse=False):
		# Reverse is true when the order of parameters is reversed. Temporariy solution.
		if not reverse:
			D,H = self._vars_to_factors(delta,eta)
			return self._likelihood(D,H) + 0.5*(delta.T @ delta) + 0.5*(eta.T @ eta)
		else:
			D,H = self._vars_to_factors(eta,delta)
			return self._likelihood(D,H) + 0.5*(delta.T @ delta) + 0.5*(eta.T @ eta)

	def gradient_eta(self,eta,delta,reverse=None): # Todo: improve temp solution
		D,H = self._vars_to_factors(delta,eta)

		hadamard_left = utils.vec( D.T @ (D @ H - self.X) )
		hadamard_right = self.h_link.derivative( self.Ch.T @ eta )
		return self.sigN**(-2) * (hadamard_left * hadamard_right).T @ self.Ch.T + eta

	def gradient_delta(self,delta,eta):
		D,H = self._vars_to_factors(delta,eta)

		hadamard_left = utils.vec( (D @ H - self.X) @ H.T)
		hadamard_right = self.d_link.derivative( self.Cd.T @ delta )
		
		return self.sigN**(-2) * (hadamard_left*hadamard_right).T @ self.Cd.T + delta

	def optimize(self, num_iters = 50, print_steps = 100 ):
		KM = self.K * self.M
		ML = self.M * self.L

		# Initial guess
		delta =  np.random.randn(KM)
		eta = np.random.randn(ML)

		for i in range(num_iters):
			if (i+1) % print_steps == 0:
				print("Optimzing: " + str(i+1) + '/' + str(num_iters))
			delta = minimize(self.total_loss,delta,args=(eta),method='L-BFGS-B', jac=self.gradient_delta).x

			eta = minimize(self.total_loss,eta,args=(delta,True),method='L-BFGS-B', jac=self.gradient_eta).x
			
		final_loss = self.total_loss(delta,eta)
		D, H = self._vars_to_factors(delta,eta)
		return D, H, final_loss










