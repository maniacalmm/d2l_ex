from mxnet import autograd, np, npx, gluon
from IPython import display
import matplotlib.pyplot as plt
import d2l
npx.set_np()

# basically what dataloader is a mechanism that helps to load data efficiently during the 
# training and testing period, in batches. this process can be parallelized. if the model is 
# not sequentially limited(list RNN), then the order of which these data feed into the model doesn't
# really matter that much, it can be interleaved between different processes

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

num_inputs = 784
num_outputs = 10

W = np.random.normal(0, 0.01, (num_inputs, num_outputs))
b = np.zeros(num_outputs)

W.attach_grad()
b.attach_grad()

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

# Saved in the d2l package for later use
class Animator(object):
    def __init__(self, xlabel=None, ylabel=None, legend=[], xlim=None,
                 ylim=None, xscale='linear', yscale='linear', fmts=None,
                 nrows=1, ncols=1, figsize=(3.5, 2.5)):
        """Incrementally plot multiple lines."""
        d2l.use_svg_display()
        self.fig, self.axes = d2l.plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1: self.axes = [self.axes,]
        # use a lambda to capture arguments
        self.config_axes = lambda : d2l.set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        """Add multiple data points into the figure."""
        if not hasattr(y, "__len__"): y = [y]
        n = len(y)
        if not hasattr(x, "__len__"): x = [x] * n
        if not self.X: self.X = [[] for _ in range(n)]
        if not self.Y: self.Y = [[] for _ in range(n)]
        if not self.fmts: self.fmts = ['-'] * n
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        display.display(self.fig)
        display.clear_output(wait=True)


def softmax(X):
    X_exp = np.exp(X)
    partition = X_exp.sum(axis=1, keepdims=True)
    return X_exp / partition

def net(X):
    internal = np.dot(X.reshape(-1, num_inputs), W) + b
    return softmax(internal)

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

def evaluate_accuracy(net, data_iter):
    metric = Accumulator(2)
    for X, y in data_iter:
        metric.add(accuracy(net(X), y), y.size)
    return metric[0] / metric[1]

# before training
print(evaluate_accuracy(net, test_iter))

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


def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):
    trains, test_accs = [], []
    # animator = Animator(xlabel='epoch', xlim=[1, num_epochs],
    #                     ylim=[0.3, 0.9],
    #                     legend=['train loss', 'train acc', 'test acc'])    
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        print("epoch: {}, acc: {}".format(epoch + 1, test_acc))
        # animator.add(epoch + 1, train_metrics + (test_acc,))

# num_epochs, lr = 10, 0.1
# updater = lambda batch_size: d2l.sgd([W, b], lr, batch_size)
# train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, updater)