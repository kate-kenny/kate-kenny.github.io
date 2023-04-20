import numpy as np 

class SVD: 
    def compare_images(self, A, A_):

        fig, axarr = plt.subplots(1, 2, figsize = (7, 3))

        axarr[0].imshow(A, cmap = "Greys")
        axarr[0].axis("off")
        axarr[0].set(title = "original image")

        axarr[1].imshow(A_, cmap = "Greys")
        axarr[1].axis("off")
        axarr[1].set(title = "reconstructed image")

    def svd_reconstruct(self, img, k): 
        #reconstructs img from SVD using k values

        A = img
       
        U, sigma, V = np.linalg.svd(A)
        
        #construct diagonal matrix D whose entries are entires of sigma
        D = np.zeros_like(A,dtype=float) # matrix of zeros of same shape as A
        D[:min(A.shape),:min(A.shape)] = np.diag(sigma)
        
        #index first k rows/entries/columns of U, D, and V respectively
        U_ = U[:,:k]
        D_ = D[:k, :k]
        V_ = V[:k, :]

        A_ = U_ @ D_ @ V_
        self.compare_images(A, A_)

    def svd_experiment(self, img):
        #reconstructs img with several different k's, determines storage needed 
        k = 3