import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.decomposition import NMF

if __name__ == '__main__':
	"""
	Load generated data
	"""
	data = loadmat('../../Data/gendata.mat')
	X = data['X']
	X_true = data['DD']
	spectrum = data['spectrum']

	model = NMF(n_components=2,alpha=0.0,l1_ratio=0.0)
	D = model.fit_transform(X_true)
	H = model.components_
	X_reconstruct = D @ H

	plot = True
	#upper left plot is the reconstructed matrix
	if plot:
		plt.figure()
		plt.imshow(X_reconstruct,aspect='auto')
		plt.title("Recon")
		plt.axis("off")

		plt.figure()
		plt.title('Spectrum')
		plt.plot(spectrum)

		plt.figure()
		plt.title('ML')
		plt.plot(D[:,0])

		plt.figure()
		plt.title('ML')
		plt.plot(H.T[:,0])
		
		plt.show()