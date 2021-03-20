import numpy as np
from scipy.special import logsumexp
from softmax import softmax

d = np.ones((2,2, 5))
d[:,:, 0] = np.array([[2,1], [3,4]])
d[:,:, 3] = np.array([[2,1], [3,4]])
st = 'aiiai'
a = np.where(np.array(list(st)) == 'a')[0]
print(a)
b = np.full((5), False)
print(b)
b[a] = True
print(b)
print(np.sum(d, axis=1))