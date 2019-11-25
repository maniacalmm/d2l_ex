import d2l_dx
from mxnet import autograd, gluon, np, npx, init
npx.set_np()

n_train, n_test, num_inputs, batch_size = 20, 100, 200, 1
true_w, true_b = np.ones((num_inputs, 1)) * 0.01, 0.05

train_data = d2l_dx.synthetic_data(true_w, true_b, n_train)
train_iter = d2l_dx.load_array(train_data, batch_size)

test_data = d2l_dx.synthetic_data(true_w, true_b, n_test)
test_iter = d2l_dx.load_array(test_data, batch_size)

def init_params():
    w = np.random.normal(scale=1, size=(num_inputs, 1))
    b = np.zeros(1)
    w.attach_grad()
    b.attach_grad()
    return [w, b]

def l2_penalty(w):
    return (w**2).sum() / 2

def train(lambd):
    # w and b has been 'grad attached'
    w, b = init_params()
    net, loss = lambda X: d2l_dx.linreg(X, w, b), d2l_dx.squared_loss
    num_epochs, lr = 100, 0.003

    for epoch in range(1, num_epochs + 1):
        for X, y in train_iter:
            with autograd.record():
                l = loss(net(X), y) + lambd * l2_penalty(w)
            l.backward()
            d2l_dx.sgd([w, b], lr, batch_size)
        if epoch % 5 == 0:
            train_loss = d2l_dx.evaluate_loss(net, data_iter=train_iter, loss=loss)
            test_loss = d2l_dx.evaluate_loss(net, data_iter=test_iter, loss=loss)
            print("epochs: {}, training loss: {}, test loss: {}".format(epoch, train_loss, test_loss))
    print('l1 norm of w', np.abs(w).sum())

def train_gluon(wd):
    net = gluon.nn.Sequential()
    net.add(gluon.nn.Dense(1))
    net.initialize(init.Normal(sigma=1))
    loss = gluon.loss.L2Loss()
    num_epochs, lr = 100, 0.003

    trainer_w = gluon.Trainer(net.collect_params('.*weight'), 'sgd', {'learning_rate': lr, 'wd': wd})
    trainer_b = gluon.Trainer(net.collect_params('.*bias'), 'sgd', {'learning_rate': lr})

    for epoch in range(1, num_epochs + 1):
        for X, y in train_iter:
            with autograd.record():
                l = loss(net(X), y)
            l.backward()
            trainer_w.step(batch_size)
            trainer_b.step(batch_size)
        if epoch % 5 == 0:
            train_loss = d2l_dx.evaluate_loss(net, data_iter=train_iter, loss=loss)
            test_loss = d2l_dx.evaluate_loss(net, data_iter=test_iter, loss=loss)
            print("epochs: {}, training loss: {}, test loss: {}".format(epoch, train_loss, test_loss))
    print('L1 norm of w: ', np.abs(net[0].weight.data()).sum())

train_gluon(2)