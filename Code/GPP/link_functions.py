# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 22:23:35 2018

@author: Niklas
"""

import numpy as np
import utils

class exp_to_gauss:
	"""
	Implements inverse exponential-to-Gaussian link function and its derivate.
	"""
	def __init__(self,lambd,sig):
		self.lambd = lambd
		self.sig = sig

	def inverse(self,x):
		#print('sigma:' + str(self.sig))
		#print('inner_erf:', x/(np.sqrt(2)*self.sig))
		#print('erf:',utils.erf( x/(np.sqrt(2)*self.sig) ))
		#print('inverse:',-(1/self.lambd) * np.log( 0.5 - 0.5 *  utils.erf( x/(np.sqrt(2)*self.sig) ) ))
		res = np.maximum(-(1/self.lambd) * np.log( 0.5 - 0.5 *  utils.erf( x/(np.sqrt(2)*self.sig) ) + 1e-14 ),0)
		return  np.nan_to_num(res)

	def derivative(self,x):
		coeff = (1/( np.sqrt(2*np.pi) * self.sig * self.lambd))
		expo = np.exp( self.lambd * self.inverse(x) - x**2/(2*self.sig**2) )
		return coeff * expo

	def __call__(self,x):
		return self.inverse(x) , self.derivative(x)


class rect_gauss:
	"""
	Implements inverse rectified-Gaussian-to-Gaussian link function and its derivate.
	"""
	def __init__(self,s,sig):
		self.s = s
		self.sig = sig

	def inverse(self,x):
		return np.sqrt(2) * self.s * utils.inv_erf( 0.5 + 0.5 * utils.erf( x/(np.sqrt(2)*self.sig) ) )

	def derivative(self,x):
		coeff = self.s/(2*self.sig)
		exp_1 = self.inverse(x)**2 / (2 * self.s**2)
		exp_2 = x**2 / (2 * self.sig**2)

		return coeff * np.exp(exp_1 - exp_2)

	def __call__(self,x):
		return self.inverse(x) , self.derivate(x)
