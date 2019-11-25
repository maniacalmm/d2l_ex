import d2l_dx
from mxnet import autograd, gluon, init, np, npx
from mxnet.gluon import nn
npx.set_np()

def dropout(X, drop_prob):
    assert 0 <= drop_prob <= 1
    if drop_prob == 1:
        return np.zeros_like(X)
    mask = np.random.uniform(0, 1, X.shape) > drop_prob
    return mask.astype(np.float32) * X / (1.0 - drop_prob)

X = np.arange(16).reshape(2, 8)
print(dropout(X, 0), '\n')
print(dropout(X, 0.5), '\n')
print(dropout(X, 1), '\n')

num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 784, 10, 256, 256

# three layers
W1 = np.random.normal(scale=0.01, size=(num_inputs, num_hiddens1))
b1 = np.zeros(num_hiddens1)
W2 = np.random.normal(scale=0.01, size=(num_hiddens1, num_hiddens2))
b2 = np.zeros(num_hiddens2)
W3 = np.random.normal(scale=0.01, size=(num_hiddens2, num_outputs))
b3 = np.zeros(num_outputs)

params = [W1, b2, W2, b2, W3, b3]
for param in params:
    param.attach_grad()

drop_prob1, drop_prob2 = 0.4, 0.4

def net(X):
    X = X.reshape(-1, num_inputs)
    H1 = npx.relu(np.dot(X, W1) + b1)
    if autograd.is_training():
        H1 = dropout(H1, drop_prob1)
    H2 = npx.relu(np.dot(H1, W2) + b2)
    if autograd.is_training():
        H2 = dropout(H2, drop_prob2)
    return np.dot(H2, W3) + b3

num_epochs, lr, batch_size = 30, 0.5, 256
loss = gluon.loss.SoftmaxCrossEntropyLoss()
train_iter, test_iter = d2l_dx.load_data_fashion_mnist(batch_size)
updater = lambda batch_size: d2l_dx.sgd(params, lr, batch_size)
d2l_dx.train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)
