import numpy as np
from scipy import io, integrate, linalg, signal
from scipy.sparse.linalg import cg, eigs

a=np.zeros((3,3))
print(np.ndim(a))
print(np.size(a))
print(np.shape(a))
n=1
print(a.shape[n-1])

a=np.array([[1., 2., 3.], [4., 5., 6.]])
print(a)
a=np.zeros((9,9))
print(a[-1])
print(a[1,4])
print(a[1])
print(a[0:5])
print(a[-5:])
print(a[0:3, 4:9])
print(a[np.ix_([1, 3, 4], [0, 2])])
print(a[::2, :])
print(a[2:9:2,:])
print(a[np.r_[:len(a),0]])
print(a.transpose())
print(a.conj().transpose())

b=np.ones((9,9))
v=np.ones(9)
print(a@b)
print(a*b)
print(a/b)
print(a**3)
print((a > 0.5))
print(np.nonzero(a > 0.5))
print(a[:,np.nonzero(v > 0.5)[0]])
print(a[:, v.T > 0.5])
a[a < 0.5]=0
print(a)
print(a * (a > 0.5))
a[:] = 3
print(a)

x=np.ones((9,9))
y = x.copy()
print(y)
y = x[1, :].copy()
print(y)
y = x.flatten()
print(y)

print(np.arange(1., 11.))
print(np.arange(10.))
print(np.arange(1.,11.)[:, np.newaxis])
print(np.zeros((3, 4)))
print(np.zeros((3, 4, 5)))
print(np.ones((3, 4)))
print(np.eye(3))
a=[1,2,3]
print(np.diag(a))
v=[0,1,2,3]
print(np.diag(v, 0))

from numpy.random import default_rng
rng = default_rng(42)
rng.random((3,4))
print(rng)
print(np.linspace(1,3,4))
print(np.mgrid[0:9.,0:6.])
print(np.ix_(np.r_[0:9.],np.r_[0:6.]))
print(np.meshgrid([1,2,4],[2,4,5]))
print(np.ix_([1,2,4],[2,4,5]))

b=np.ones((9,9))
a=np.ones((9,9))

print(np.tile(a, (3, 2)))
print(np.hstack((a,b)))
print(np.vstack((a,b)))
print(a.max())
print(a.max(0))
print(	a.max(1))
print(np.maximum(a, b))

v=np.ones(3)
print(np.sqrt(v @ v))
print(np.logical_and(a,b))
print(np.logical_or(a,b))
a=np.array([[1,2],[3,4]])
print(linalg.inv(a))
print(linalg.pinv(a))
print(np.linalg.matrix_rank(a))
b=np.array([5,6])
print(linalg.solve(a, b))
a=np.array([True,False,True])
b=np.array([True,True,True])
print(a & b)
print(a | b)


a=np.array([[2,2],[3,4]])
print(linalg.svd(a))
print(linalg.cholesky(a))
a=np.ones((9,9))
b=np.ones((9,9))
print( linalg.eig(a))
print(linalg.eig(a, b))
print(eigs(a, k=3))
print(linalg.qr(a))
print(linalg.lu(a))

print(np.fft.fft(a))
print(np.fft.ifft(a))
print(np.sort(a))
print(np.sort(a, axis=1))
print(np.argsort(a[:, 0]))
Z=a
y=np.ones(9)
print(linalg.lstsq(Z, y))
x=np.array([1,2,1,2,1,2,1,2,])
print(signal.resample(x, 2))
print(np.unique(a))
print(a.squeeze())



