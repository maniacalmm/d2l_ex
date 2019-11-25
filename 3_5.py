import matplotlib.pyplot as plt
import d2l
from mxnet import gluon
import sys
# d2l.use_svg_display()

mnist_train = gluon.data.vision.FashionMNIST(train=True)
mnist_test = gluon.data.vision.FashionMNIST(train=False)

print(len(mnist_train), len(mnist_test))

def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress',
                    'coat', 'sandal', 'shirt', 'sneaker', 'bag',
                    'ankle boot']
    return [text_labels[int(i)] for i in labels]

def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
    """
    plot a list of images
    """
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)

    def set_axis_invisible(ax):
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        return ax
        
    axes = map(set_axis_invisible, axes.flatten())
    # axes.fl
    # _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)
    # axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        ax.imshow(img.asnumpy())
        if titles:
            ax.set_title(titles[i])
    return axes

# X, y = mnist_train[:10]
# show_images(X.squeeze(axis=-1), 2, 5, titles=get_fashion_mnist_labels(y))
# plt.show()

def get_dataloader_workers(num_works=4):
    if sys.platform.startswith('win'):
        return 0
    else:
        return num_works

batch_size = 256
transformer = gluon.data.vision.transforms.ToTensor()
train_iter = gluon.data.DataLoader(mnist_train.transform_first(transformer),
                                    batch_size, 
                                    shuffle=True, 
                                    num_workers=get_dataloader_workers(2))

timer = d2l.Timer()
for X, y in train_iter:
    continue
print("{:.2f} seconds".format(timer.stop()))

def load_data_fashion_mnist(batch_size, resize=None):
    # form transformation
    # Resize -> ToTensor
    dataset = gluon.data.vision
    trans = [dataset.transforms.Resize(resize)] if resize else []
    trans.append(dataset.transforms.ToTensor())
    trans = dataset.transforms.Compose(trans)

    #prepare dataset & transform
    mnist_train = dataset.FashionMNIST(train=True).transform_first(trans)
    mnist_test = dataset.FashionMNIST(train=False).transform_first(trans)

    # return dataloader
    return (gluon.data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=get_dataloader_workers()),
            gluon.data.DataLoader(mnist_test, batch_size, shuffle=False, num_workers=get_dataloader_workers()))

# load_data_fashion_mnist(32, (64, 64))
# train_iter, test_iter = load_data_fashion_mnist(32, (64, 64))
# for X, y in train_iter:
#     print(X.shape)
#     break