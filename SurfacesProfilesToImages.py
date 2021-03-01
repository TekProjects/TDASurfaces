### python > 3.6
### numpy, scipy, ripser, persim, matplotlib
### Tarek FRAHI
### MIT License

import numpy as np
from scipy import sparse
from ripser import ripser
from persim import PersImage
import matplotlib.pyplot as plt

class SurfacesProfilesToImages:
    '''
    Author: Tarek FRAHI
            tarek.frahi@gmail.com
    Ref: Tape surfaces characterization with persistence images (2020)
         https://www.aimspress.com/article/doi/10.3934/matersci.2020.4.364
    '''
    def __init__(self,thresh,pixels):
        '''
        Transform 1D surface profiles to persistence images
        Params:
            thresh: threshold persistence for diagrams
            pixels: tuple of 2 determining the persistence images pixels resolution
        '''
        self.thresh = thresh
        self.pixels = pixels
        
    def filtration(self,data):
        '''
        Compute the sublevelset filtration
        Params:
            data: list of 1D numpy arrays
        Returns a list of sparse distance matrices
        '''
        N = data.size
        I = np.arange(N-1)
        J = np.arange(1, N)
        V = np.maximum(data[0:-1], data[1::])
        I = np.concatenate((I, np.arange(N)))
        J = np.concatenate((J, np.arange(N)))
        V = np.concatenate((V, data))
        D = sparse.coo_matrix((V, (I, J)), shape=(N, N)).tocsr()

        return D
    
    def fit_transform(self,data):
        '''
        Compute the persistence images 
        Params:
            data: list of 1D numpy arrays, representing the surface profiles
        Returns a list of persistence images
        '''
        Filt = [self.filtration(d) for d in data]
        Diags = [ripser(f, maxdim=0, thresh=self.thresh, distance_matrix=True)['dgms'][0][:-1] for f in Filt]
        Imgs = PersImage(pixels=self.pixels,verbose=False).transform(Diags)
            
        return Imgs
    
## Use example (see Ref)
# S2I = SurfacesProfilesToImages(thresh=20,pixels=(20,20))
# s = np.array([11, 14, 9, 7, 9, 7, 8, 10, 9])
# data = [s]
# im = S2I.fit_transform(data)
# plt.imshow(np.flip(im[0],axis=1))