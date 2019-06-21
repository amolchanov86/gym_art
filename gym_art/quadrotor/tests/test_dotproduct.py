from scipy.linalg import get_blas_funcs
import time
import numpy as np
st = time.time()
X = np.random.randn(10, 4)
Y = np.random.randn(7, 4).T
gemms = get_blas_funcs("gemm", [X, Y])
print("Time comparison of blas",gemms(1,X,Y))
print(time.time()-st)

st = time.time()

print("Time comparison of np.dot",np.dot(X, Y)
)
print(time.time()-st)
