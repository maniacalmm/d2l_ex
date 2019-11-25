import d2l_dx
from mxnet import np, npx
npx.set_np()

def corr2d_multi_in(X, K):
    return sum(d2l_dx.corr2d(x, k) for x, k in zip(X, K))

def corr2d_multi_in_out(X, K):
    return np.stack([corr2d_multi_in(X, k) for k in K])

X = np.array([[[0, 1, 2], [3, 4, 5], [6, 7, 8]],
              [[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
K = np.array([[[0, 1], [2, 3]], [[1, 2], [3, 4]]])

print('X.shape: ', X.shape, ' K.shape: ', K.shape)

print(corr2d_multi_in(X, K).shape)

K_stacked = np.stack((K, K + 1, K + 2))
print('K_stacked.shape: ', K_stacked.shape)

print(corr2d_multi_in_out(X, K_stacked).shape)


