import d2l 
from mxnet import np, npx, gluon
npx.set_np()
import d2l_dx

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)

num_inputs, num_outputs, num_hiddens = 784, 10, 256

W1 = np.random.normal(scale=0.01, size=(num_inputs, num_hiddens))
b1 = np.zeros(num_hiddens)

W2 = np.random.normal(scale=0.01, size=(num_hiddens, num_outputs))
b2 = np.zeros(num_outputs)
params = [W1, b1, W2, b2]

for param in params:
    param.attach_grad()

def relu(X):
    return np.maximum(X, 0)

def net(X):
    """
    two layers of fully connected NN
    """
    X = X.reshape(-1, num_inputs)
    H = relu(np.dot(X, W1) + b1)
    return np.dot(H, W2) + b2

loss = gluon.loss.SoftmaxCrossEntropyLoss()

num_epochs, lr = 10, 0.01
updater = lambda batch_size: d2l.sgd(params, lr, batch_size)
d2l_dx.train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)
