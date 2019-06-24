# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 23:02:01 2018

@author: Niklas

The toy example from the paper.
Used to test and demonstrate the implementation of the GPP-NMF
"""
import numpy as np
import matplotlib.pyplot as plt

import utils
import covariance_functions as cvf
import link_functions as lf

import Loss

np.random.seed(125)

### Parameters specifying dimensions
"""
Model set-up
"""
M = 2
kernel = cvf.rbf
d_link = lf.rect_gauss(s=1,sig=1)
h_link = lf.exp_to_gauss(lambd=1,sig=1)
# Kernel parameters
d_parm = 0.001
h_parm = 0.01

"""
Generate data for toy example
"""
K = 100
L = 200
X_shape = (K,L)
D_shape = (K,M)
H_shape = (M,L)

# Convenience products
KM = K*M
ML = M*L

### Generate covariance matrices for d and h
# Initialize
d_cov = np.zeros( (KM , KM) )
h_cov = np.zeros( (ML , ML) )

# Noise to prevent diagonal of zeros
eps = 1e-10

# Generate covariance matrix of d
for i in range(KM):
	for j in range(KM):
		d_cov[i,j] = kernel(i+1,j+1,d_parm)

d_cov += eps*np.eye(KM)

# Generate covariance matrix of h
for i in range(ML):
	for j in range(ML):
		h_cov[i,j] = kernel(i+1,j+1,h_parm)

h_cov += eps*np.eye(ML)


### Generate D and H
# Get D
d = np.random.randn(KM,1)
Cd = utils.cholesky(d_cov)
d_inv_link =  d_link.inverse(Cd.T @ d)
D_gen = utils.vec_inv(d_inv_link, D_shape )

# Get H
h = np.random.randn(ML,1)
Ch = utils.cholesky(h_cov)
h_inv_link = h_link.inverse(Ch.T @ h)
H_gen = utils.vec_inv(h_inv_link, H_shape)

X_gen = D_gen @ H_gen
X_gen_noisy = X_gen + 10*np.random.randn(K,L)


### Plot generated data
plt.figure()
plt.subplot(2,2,1)
plt.title('Generated X')
plt.imshow(X_gen)

plt.subplot(2,2,2)
plt.title('Generated X with noise')
plt.imshow(X_gen_noisy)


"""
Define the model
"""
loss = Loss.Loss(X_gen_noisy,M=M,d_link=d_link,h_link=h_link,sigN=7,kernel=kernel,cov_pars=(d_parm,h_parm))

"""
Run model
"""
D_opt, H_opt, final_loss = loss.optimize(num_iters=50,print_steps=10)
X_opt = D_opt @ H_opt

print("Loss: {}".format(final_loss))

plt.subplot(2,2,3)
plt.title('GPP-NMF Constructed X')
plt.imshow(X_opt)

plt.figure()
plt.subplot(1,2,1)
plt.title('Generated D')
plt.imshow(D_gen,aspect='auto')

plt.subplot(1,2,2)
plt.title('GPP-NMF Constructed D')
plt.imshow(D_opt,aspect='auto')

plt.figure()
plt.subplot(1,2,1)
plt.title('Generated H')
plt.imshow(H_gen,aspect='auto')

plt.subplot(1,2,2)
plt.title('GPP-NMF Constructed H')
plt.imshow(H_opt,aspect='auto')

### Plot generated data
plt.show()