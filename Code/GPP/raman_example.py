import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

import utils
import covariance_functions as cvf
import link_functions as lf

import Loss


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
    M = 2
    kernel = cvf.laplacian
    h_link = lf.exp_to_gauss(lambd=1, sig=1)
    d_link = lf.rect_gauss(s=1, sig=1)
    # Kernel parameters
    d_parm = 1
    h_parm = 10 

    """
	Define the model
	"""
    loss = Loss.Loss(X, M=M, d_link=d_link, h_link=h_link,
                     sigN=2, kernel=kernel, cov_pars=(d_parm, h_parm))

    """
	Run model
	"""
    D, H, final_loss = loss.optimize(num_iters=50, print_steps=10)
    X_reconstruct = D @ H

    print("Loss: {}".format(final_loss))

    plot = True
    # upper left plot is the reconstructed matrix
    if plot:
        plt.figure()
        plt.imshow(X_reconstruct, aspect='auto')
        plt.title("Recon")
        plt.axis("off")
        plt.savefig("gpp_lap_X.eps", bbox_inches="tight")
        
        # lower left plot is the original matrix
        plt.figure()
        plt.imshow(X,aspect='auto')
        plt.title("Noise")
        plt.axis("off")
        #plt.savefig("noisy_X.eps", bbox_inches="tight")

        plt.figure()
        plt.imshow(X_true,aspect='auto')
        plt.title("True")
        plt.savefig("true_X.eps", bbox_inches="tight")

        #plt.figure()
        #plt.title('GPP')
        #plt.plot(D[:, 0])
        #plt.axis("off")

        plt.figure()
        plt.title('GPP')
        plt.plot(H.T[:, 0])
        plt.savefig("gpp_lap_spectrum.eps", bbox_inches="tight")

        plt.show()
