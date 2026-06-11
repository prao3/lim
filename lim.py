import numpy as np
import scipy.linalg as lg


class LIM():
    def __init__(self, x, tau0:int, L=None):
        '''
        A class to compute a linear inverse model

        Parameters
        ----------
        x (np.array): data to compute LIM with
        tau0 (int)  : Time lag to use to compute LIM
        L (np.array): Optional pre-computed LIM. If given, skips computation of LIM
        '''
        # If no L is given, compute L
        if L == None:
            self.L = self.__calc_L__(x, tau0)
        # Otherwise, just keep given L
        else:
            self.L = L

        # Calculating eigenvalues and eigenvectors
        w, vl, vr = lg.eig(self.L, left=True)
        # Sorting by magnitude, largest eigenvalue first
        sortindex = np.argsort(np.abs(w))
        w = w[sortindex][::-1]
        vr = vr[:, sortindex][:, ::-1]
        vl = vl[:, sortindex][:, ::-1]
        # Rotating eigenvectors and storing them
        vr = self.__rotate_vector__(vr)
        # Normalizing adjoints so vl[:, i] @ vr[:, j] = delta(i, j)
        vl = vl / np.sqrt(self.__vmat_dot__(vr, np.conj(vr)))

        self.w = w
        self.vr = vr
        self.vl = vl
    
    def getSystemMatrix(self):
        '''
        Returns system matrix L
        '''
        return self.L
    
    def getEigenvectors(self):
        '''
        Returns eigenvectors of L
        '''
        return self.vr
    
    def getAdjoints(self):
        '''
        Returns adjoints of L
        '''
        return self.vl
    
    def getEigenvalues(self):
        '''
        Returns eigenvalues of L
        '''
        return self.w
    
    def __rotate_vector__(self, v):
        '''
        Rotates a given array of vectors, such that the real part is orthogonal to the imaginary part of the vector.
        If a = real(v) and b = imag(v), then the rotated vector satisfies
        a dot a = 1, b dot b > 1, a dot b = 0
        
        Parameters
        ----------
        v (np.array): Array of vectors to rotate, where v[:, i] is the ith vector

        Returns
        -------
        np.array: Rotated vector array
        '''
        # Calculating magnitude of vector
        dotprod = self.__vmat_dot__(v, v)
        real = np.real(dotprod)
        imag = np.imag(dotprod)

        # Calculating angle to rotate by
        theta = np.arctan(imag / real)
        # Rotating vector
        v = v * np.exp(-0.5j * theta)[np.newaxis, :]

        # Calculating imaginary and real part magnitudes
        imag_prod = self.__vmat_dot__(np.imag(v), np.imag(v))
        real_prod = self.__vmat_dot__(np.real(v), np.real(v))

        # If imaginary product is smaller than real product
        # AND the imaginary product is non zero
        # Swap real and imaginary part of v
        mask = (imag_prod < real_prod) * (imag_prod != 0)
        v[:, mask] = np.real(v[:, mask]) * 1j + np.imag(v[:, mask])

        # Normalizing by real part
        v = v / np.sqrt(self.__vmat_dot__(np.real(v), np.real(v)))
        return v
    
    def __vmat_dot__(self, a, b):
        '''
        Dot product of an array of vectors

        Parameters
        ----------
        a (np.array): Array of vectors, where a[:, i] is the ith vector
        b (np.array): Second array of vectors, where b[:, i] is the ith vector

        Returns
        -------
        np.array: Array of dot product results, where res[i] is the ith product
        '''
        return np.sum(a * b, axis=0)

    def __calc_L__(self, x, tau0):
        '''
        Calculates system matrix L from state vector x and lag tau0

        Parameters
        ----------
        x: np.array of shape (m, n), where n is the number of samples and m is the number of variables
        tau0: int, lag to use to calculate LIM

        Returns
        -------
        np.array: System matrix of shape (m, m)
        '''
        # Covariance matrix
        cov = self.__covariance__(x, x)
        # Lag covariance
        lag_cov = self.__covariance__(x[:, tau0:], x[:, :-tau0])
        # Calculating G
        G = lag_cov @ lg.inv(cov)
        # Calculating L
        L = 1/tau0 * lg.logm(G)

        # Calculating error on log G
        # (from scipy documentation)
        err = lg.norm(lg.expm(lg.logm(G)) - G, 1) / lg.norm(G, 1)
        # If L is real within error, make it real
        L = np.real_if_close(L, err)

        # Returning L
        return L
    
    def __covariance__(self, x, y):
        '''
        Calculates the covariance matrix between samples x and y

        Parameters
        ----------
        x: np.array of shape (m, n), where n is the number of samples and m is the number of variables
        y: np.array of shape (m, n)

        Returns
        -------
        np.array: shape (m, m) covariance matrix
        '''
        # Degrees of freedom for normalization
        n = x.shape[-1]
        # Subtracting mean
        x = x - x.mean(axis=1)[:, np.newaxis]
        y = y - y.mean(axis=1)[:, np.newaxis]
        # Returning covariance matrix
        return np.dot(x, y.transpose()) / (n - 1)