from mxnet import np, npx, autograd
import matplotlib.pyplot as plt
import random
npx.set_np()

def synthetic_data(w, b, num_examples):
    """
    generate y = X w + b + noise
    """
    X = np.random.normal(0, 1, (num_examples, len(w)))
    y = np.dot(X, w) + b
    y += np.random.normal(0, 0.01, y.shape)
    return X, y

def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    # The examples are read at random, in no particular order
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = np.array(indices[i: min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]


batch_size = 10
true_w = np.array([2, -3.4, 12, 20.28])
true_b = 7.3

features, labels = synthetic_data(true_w, true_b, 1000)

w = np.random.normal(0, 0.01, true_w.shape)
b = np.zeros(1)

w.attach_grad()
b.attach_grad()

def linreg(X, w, b):
    return np.dot(X, w) + b

def squared_loss(y_hat, y):
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / len(y_hat)

def sgd(params, lr, batch_size):
    for param in params:
        param[:] -= lr * param.grad / batch_size

lr = 0.03
num_epochs = 10
net = linreg
loss = squared_loss

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        with autograd.record():
            l = loss(net(X, w, b), y)
        l.backward()
        sgd([w, b], lr, batch_size)
    train_l = loss(net(features, w, b), labels)
    print('epoch: {}, loss: {}'.format(epoch + 1, train_l.mean().asnumpy()))

print('true_w: {}, true_b: {}'.format(true_w, true_b))
print(w, b)