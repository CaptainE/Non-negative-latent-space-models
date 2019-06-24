# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 22:23:54 2018

@author: Niklas
"""
import numpy as np
import scipy.special as sp

def rbf(x,y,gamma):
	"""
	The Gaussian radial basis function (RBF)
	Params:
		x and y:   sample indices
		gamma:     length-scale, determines smoothness of the factors
	"""
	return np.exp( -gamma*( (x-y)**2) )

def cauchy(x,y,sigma):
	"""
	The Cauchy kernel
	Params:
		x and y:   sample indices
		sigma:     scaling parameter
	"""
	d = x-y
	return 1.0/(1.0+((d**2)/sigma**2))

def matern(x,y,parms):
	"""
	The Matern kernel
	Params:
		x and y:	sample indices
		parms:		kernel params (sigma,nu,rho)
	"""
	sigma,nu,rho = parms
	d = np.abs(x-y)
	z = np.sqrt(2*nu)*(d/rho)
	return sigma**2*((2**(1-nu))/sp.gamma(nu))*z**nu*sp.kv(nu,z)

def laplacian(x,y,sigma):
	"""
	The Laplacian kernel
	"""
	return np.exp( - ((x-y)**2) / sigma)