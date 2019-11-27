import d2l_dx
from mxnet import autograd, np, npx, gluon, init
from mxnet.gluon import nn
import matplotlib.pyplot as plt
npx.set_np()

T = 1000
time = np.arange(0, T)
x = np.sin(0.01 * time) + 0.2 * np.random.normal(size=T)

# plt.plot(x)
# plt.show()

tau = 4
features = np.zeros((T - tau, tau))

for i in range(tau):
    features[:, i] = x[i: T - tau + i]
labels = x[tau:]

