import d2l_dx
from mxnet import autograd, gluon, np, npx
from mxnet.gluon import nn
npx.set_np()

maxdegree = 20
n_train, n_test = 100, 100
true_w = np.zeros(maxdegree)
true_w[0:4] = np.array([5.0, 1.2, -3.4, 5.6])

features = np.random.normal(size=(n_train + n_test, 1))
features = np.random.shuffle(features)
power = np.arange(maxdegree).reshape(1, -1)
poly_features = np.power(features, power)
poly_features = poly_features / (npx.gamma(np.arange(maxdegree) + 1).reshape(1, -1))
labels = np.dot(poly_features, true_w)
labels += np.random.normal(scale=0.1, size=labels.shape)

print(poly_features.shape, true_w.shape, labels.shape)
print("----------------------")
print(poly_features[0], true_w)
print("----------------------")
print(features[0])
print("----------------------")
print(poly_features[0])
print("----------------------")
print(labels[0])

