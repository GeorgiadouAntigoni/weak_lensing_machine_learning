"""
Manifold Learning techniques for dimensional reduction
of power spectrum data.

@author - Janu Verma
j.verma5@gmail.com
"""

import sys
import numpy as np 
from sklearn import manifold
from sklearn import decomposition



class DimensionalReduction:
	"""
	Implements manifold learning dimensional reduction techniques.
	"""
	def __init__(self, input_data):
		"""
		Parameters
		----------
		input_data: A (n,m) numpy array containing the input data matrix.
		"""
		self.input = input_data



	def TSVD(self, nComp):
		"""
		Computes the Singular Value Decomposition for PCA.

		Parameters
		----------
		nComp: Number of projection vectors. Dimension of the reduced space.


		Returns
		-------
		(n, nComp) dimensional numpy array containing reduced data.
		"""
		model = decomposition.TruncatedSVD(n_components=nComp)
		return model.fit_transform(self.input)


	def PCA(self):
		model = decomposition.PCA()
		X = model.fit_transform(self.input)
		print model.explained_variance_ratio_
		return X



	def MDS(self, nComp, n_init=1, max_iter=100):
		"""
		Computes the Multidimensional Scaling.

		Parameters
		----------
		n_init: Initial value. Default is 1.
		max_iter: Number of iterations. Default is 100.
		nComp: Number of projection vectors. Dimension of the reduced space.


		Returns
		-------
		(n, nComp) dimensional numpy array containing reduced data.
		"""
		model = manifold.MDS(n_components=nComp, n_init=n_init, max_iter=max_iter)
		return model.fit_transform(self.input)



	def ISOMAP(self, nComp, nNeighbors):
		"""
		Computes ISOMAP manifold learning technique.

		Parameters
		----------
		nComp: Number of projection vectors. Dimension of the reduced space.
		nNeighbors: Number of neighbors, required for all manifold learning,

		Returns
		-------
		(n, nComp) dimensional numpy array containing reduced data.
		"""
		model = manifold.Isomap(n_neighbors=nNeighbors, n_components=nComp)
		return model.fit_transform(self.input)


	def LLE(self, nComp, nNeighbors, method='standard'):
		"""
		Computes Locally linear embedding technique.

		Parameters
		----------
		nComp: Number of projection vectors. Dimension of the reduced space.
		nNeighbors: Number of neighbors, required for all manifold learning,
		method: Default is standard. Other choices are modified, hessian, ltsa.

		Returns
		-------
		(n, nComp) dimensional numpy array containing reduced data.
		"""
		model = manifold.LocallyLinearEmbedding(n_neighbors=nNeighbors, n_components=nComp, method=method)
		return model.fit_transform(self.input)


	def SpectralEmbedding(self, nComp):
		"""
		Computes Spectral Embedding algorithm.

		Parameters
		----------
		nComp: Number of projection vectors. Dimension of the reduced space.


		Returns
		-------
		(n, nComp) dimensional numpy array containing reduced data.
		"""
		model = manifold.SpectralEmbedding(n_components=nComp, random_state=0, eigen_solver='arpack')
		return model.fit_transform(self.input)



	def tSNE(self, nComp):
		"""
		Computes the t-Stochastic Neighborhood Embedding.

		Parameters
		----------
		nComp: Number of projection vectors. Dimension of the reduced space.


		Returns
		-------
		(n, nComp) dimensional numpy array containing reduced data.
		"""
		model = manifold.TSNE(n_components=nComp, init='pca', random_state=0)
		return model.fit_transform(self.input)



	def predict(self, testData, model):
		"""
		Computes the dimensioanlly reduced value for test set.

		Parameters
		----------
		testData: A (n,m) numpy array.
		model: Chioce of model to be used.

		Returns
		-------
		A numpy array of (n, nComp) shape. 
		"""
		return model.predict(testData)


