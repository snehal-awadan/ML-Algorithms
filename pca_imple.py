'''
What is PCA:

Principal Component Analysis is a popular unsupervised algorithm used for dimensionality reduction. 
It helps to reduce the number of feature in your dataset by combining the features without losing too much information.
It finds a linear data transformation that projet the data into a new coordinate system with a fewer dimensions.
To capture the most variation in the original data, this projection is done by finding the so-called principal components - eigenvectors of the data's covariance matrix - and multiplying the actual data matrix with a subset of the components. 
 
'''


import numpy as np


class PCA:

    '''
    the __init__ method runs once when the initialize the PCA class object.

    where,
    self.components: array with the principal component weights
    self.mean: mean variable values observed in the training data
    self.variance_share: proportion of variance explained by principal components

'''

    def __init__(self, num_components):
        self.num_components = num_components
        self.components = None
        self.mean = None
        self.variance_share = None

    '''
    This method will be applied to a provided dataset to identfiy and memorize principal components.
    '''
    def fit(self, X):
        '''
        Find Principlal Components
        '''
        # data centering:
        self.mean = np.mean(X, axis=0)
        X = X - self.mean

        # eigen values and vectors:
        cov_matrix = np.cov(X.T)
        values, vectors = np.linalg.eig(cov_matrix)

        # sort eigen values and vectors:
        sort_idx = np.argsort(values)[::-1]
        values = values[sort_idx]
        vectors = vectors[:, sort_idx]

        # store princlipal components and variance:
        self.components = vectors[:, :self.num_components]
        self.variance_share = np.sum(values[:self.num_components]) / np.sum(values)

    def transform(self, X):
        '''
        Transform new data
        '''
        # data centering:
        X = X - self.mean

        # decomposition:
        return np.dot(X, self.components)
    
# test the implementation:
X_old = np.random.normal(loc = 0, scale = 1, size = (1000, 10))
X_new = np.random.normal(loc = 0, scale = 1, size = (500, 10))

print(X_old.shape, X_new.shape)

# initialize PCA:
pca =  PCA(num_components=8)

# fit PCA on old data:
pca.fit(X_old)

# check explained variance:
print(f"Variance explained: {pca.variance_share:.4f}")

# transform datasets
X_old = pca.transform(X_old)
X_new = pca.transform(X_new)

print(X_old.shape, X_new.shape)