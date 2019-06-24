# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 21:55:44 2018

@author: Niklas

General utility functions
"""

import numpy as np
import scipy.special as sp

def vec(X):
	"""
	Vectorizes the given matrix by stacking rows from left to right
	"""
	# Flatten
	X_flat = X.flatten()
	# Convert column vector to row vector
	return X_flat

def vec_inv(x,shape):
	"""
	Maps a vector into a matrix of the given shape
	"""
	return np.reshape(x,shape)

def cholesky(X):
	"""
	Computes the Cholesky decomposition of a Hermitian positive-definite matrix X
	"""
	return np.linalg.cholesky(X)

def erf(x):
	"""
	Evaluates the error function on the given value x
	"""
	return sp.erf(x)

def inv_erf(x):
	"""
	Evaluates the inverse error function the given value x
	"""
	return sp.erfinv(x)