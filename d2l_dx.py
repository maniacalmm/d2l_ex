from mxnet import autograd, np, npx, gluon, context, init
import time
from IPython import display
import matplotlib.pyplot as plt
import sys
npx.set_np()

# accuracy is a valid concept only for classification


def linreg(X, w, b):
    return np.dot(X, w) + b

def squared_loss(y_hat, y):
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

def l2_penalty(w):
    return (w**2).sum() / 2

def synthetic_data(w, b, num_examples) -> tuple:
    """
    generate y = X w + b + noise
    """
    X = np.random.normal(0, 1, (num_examples, len(w)))
    y = np.dot(X, w) + b
    y += np.random.normal(0, 0.01, y.shape)
    return X, y

# basically what dataloader is a mechanism that helps to load data efficiently during the 
# training and testing period, in batches. this process can be parallelized. if the model is 
# not sequentially limited(list RNN), then the order of which these data feed into the model doesn't
# really matter that much, it can be interleaved between different processes

class Accumulator(object):
    """ Sum a list of numbers over time """
    def __init__(self, n):
        self.data = [0.0] * n
    def add(self, *args):
        self.data = [a + b for a, b in zip(self.data, args)] 
    def reset(self):
        self.data = [0] * len(self.data)
    def __getitem__(self, i):
        return self.data[i]

def softmax(X):
    X_exp = np.exp(X)
    partition = X_exp.sum(axis=1, keepdims=True)
    return X_exp / partition

def cross_entropy(y_hat, y):
    # picking here is an implicit multiplying, since all other entry in y are 0
    # the point to take here is that y is an array with only 0 and 1, and only
    # one of them is 1, basically a masking array
    return - np.log(y_hat[range(len(y_hat)), y])

def accuracy(y_hat, y):
    """ accuracy returns the correct number of prediction """
    if y_hat.shape[1] > 1:
        return float((y_hat.argmax(axis=1) == y.astype('float32')).sum())
    else:
        return float((y_hat.astype('int32') == y.astype('int32')).sum())

def evaluate_accuracy_gpu(net, data_iter, ctx=None):
    if not ctx:
        ctx = list(net.collect_params().values())[0].list_ctx()[0]
    metric = Accumulator(2)
    for X, y in data_iter:
        # setting the context is to move the data from cpu into GPU
        # pretty brilliant, no?
        X, y = X.as_in_context(ctx), y.as_in_context(ctx)
        metric.add(accuracy(net(X), y), y.size)
    return metric[0] / metric[1]

def evaluate_accuracy(net, data_iter):
    metric = Accumulator(2)
    for X, y in data_iter:
        metric.add(accuracy(net(X), y), y.size)
    return metric[0] / metric[1]

def evaluate_loss(net, data_iter, loss):
    metric = Accumulator(2)
    for X, y in data_iter:
        metric.add(loss(net(X), y).sum(), y.size)
    return metric[0] / metric[1]

def train_epoch_ch3(net, train_iter, loss, updater):
    metric = Accumulator(3) #train_loss_sum, train_acc_sum, num_examples
    if isinstance(updater, gluon.Trainer):
        updater = updater.step
    for X, y in train_iter:
        # compute gradients and update paramteres
        with autograd.record():
            y_hat = net(X)
            l = loss(y_hat, y)
        l.backward()
        updater(X.shape[0])
        metric.add(float(l.sum()), accuracy(y_hat, y), y.size)
    return metric[0] / metric[2], metric[1] / metric[2]

def load_array(data_arrays, batch_size, is_train=True):
    """Construct a Gluon data loader"""
    dataset = gluon.data.ArrayDataset(*data_arrays)
    return gluon.data.DataLoader(dataset, batch_size, shuffle=is_train)

def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):
    trains, test_accs = [], []
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        print("epoch: {}, acc: {}".format(epoch + 1, test_acc))

def sgd(params, lr, batch_size):
    for param in params:
        param[:] = param - lr * param.grad / batch_size

# Defined in file: ./chapter_linear-networks/fashion-mnist.md
def load_data_fashion_mnist(batch_size, resize=None):
    """Download the Fashion-MNIST dataset and then load into memory."""
    dataset = gluon.data.vision
    trans = [dataset.transforms.Resize(resize)] if resize else []
    trans.append(dataset.transforms.ToTensor())
    trans = dataset.transforms.Compose(trans)
    mnist_train = dataset.FashionMNIST(train=True).transform_first(trans)
    mnist_test = dataset.FashionMNIST(train=False).transform_first(trans)
    return (gluon.data.DataLoader(mnist_train, batch_size, shuffle=True,
                                  num_workers=get_dataloader_workers()),
            gluon.data.DataLoader(mnist_test, batch_size, shuffle=False,
                                  num_workers=get_dataloader_workers()))

def get_dataloader_workers(num_workers=4):
    # 0 means no additional process is used to speed up the reading of data.
    if sys.platform.startswith('win'):
        return 0
    else:
        return num_workers

def corr2d(X, K):
    h, w = K.shape
    Y = np.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i: i + h, j: j + w] * K).sum()
    return Y

# Defined in file: ./chapter_deep-learning-computation/use-gpu.md
def try_gpu(i=0):
    """Return gpu(i) if exists, otherwise return cpu()."""
    return context.gpu(i) if context.num_gpus() >= i + 1 else context.cpu()

class Timer:
    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        self.start_time = time.time()

    def stop(self):
        duration = time.time() - self.start_time
        self.times.append(duration)
        return duration

    def sum(self):
        return sum(self.times)

def train_ch5(net, train_iter, test_iter, num_epochs, lr, ctx=try_gpu()):
    net.initialize(force_reinit=True, ctx=ctx, init=init.Normal())
    # this is just for classification task
    loss = gluon.loss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
    timer = Timer()
    for epoch in range(num_epochs):
        metric = Accumulator(3)
        for i, (X, y) in enumerate(train_iter):
            timer.start()
            X, y = X.as_in_context(ctx), y.as_in_context(ctx)
            with autograd.record():
                y_hat = net(X)
                l = loss(y_hat, y)
            l.backward()
            trainer.step(X.shape[0])

            metric.add(l.sum(), accuracy(y_hat, y), X.shape[0])
            timer.stop()
            train_loss, train_acc = metric[0] / metric[2], metric[1] / metric[2]
            if (i + 1) % 50 == 0:
                print('i: {} ---- loss: {}, train_acc: {}'.format(i, train_loss, train_acc))
        test_acc = evaluate_accuracy_gpu(net, test_iter)
        print('loss: {}, train_acc: {}, test_acc: {}'.format(train_loss, train_acc, test_acc))
    print("-------------------------")
    print('{} examples/sec on {}'.format(metric[2] * num_epochs/timer.sum(), ctx))



