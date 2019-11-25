import d2l_dx
from mxnet import autograd, gluon, init, np, npx
from mxnet.gluon import nn
npx.set_np()

net = nn.Sequential()
net.add(
    nn.Conv2D(channels=6, kernel_size=5, padding=2, activation='sigmoid'),
    nn.AvgPool2D(pool_size=2, strides=2),
    nn.Conv2D(channels=16, kernel_size=5, activation='sigmoid'),
    nn.AvgPool2D(pool_size=2, strides=2),
    nn.Dense(120, activation='sigmoid'),
    nn.Dense(84, activation='sigmoid'),
    nn.Dense(10)
)

# X = np.random.uniform(size=(1, 1, 28, 28))
# net.initialize()
# for layer in net:
#     X = layer(X)
#     print(layer.name, 'output shape: ', X.shape)

batch_size = 256
import d2l
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

lr, num_epochs = 1, 10
d2l_dx.train_ch5(net, train_iter, test_iter, num_epochs, lr)
