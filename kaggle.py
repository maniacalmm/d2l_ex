import d2l_dx
from mxnet import autograd, gluon, init, np, npx
from mxnet.gluon import nn
import pandas as pd
import sys
from sklearn.ensemble import RandomForestRegressor
npx.set_np()

# train_data has final column as price
train_data = pd.read_csv("./data/train.csv")
# test_data doesn't has the price column
test_data = pd.read_csv("./data/test.csv")

print(train_data.shape)
print(test_data.shape)

# eliminating id information
all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]), axis=0)
print(all_features.shape)
print(all_features.dtypes)

# normalization for numeric values
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
print(numeric_features)
zero_mean_normalizer = lambda x: (x - x.mean()) / (x.std())
all_features[numeric_features] = all_features[numeric_features].apply(zero_mean_normalizer)
all_features[numeric_features] = all_features[numeric_features].fillna(0)

# categorical variables
all_features = pd.get_dummies(all_features, dummy_na=True)

n_train = train_data.shape[0]
train_features = np.array(all_features[:n_train].values, dtype=np.float32)
test_features = np.array(all_features[n_train:].values, dtype=np.float32)
train_labels = np.array(train_data.SalePrice.values, dtype=np.float32).reshape(-1, 1)

# baseline, linear model
loss = gluon.loss.L2Loss()

def get_net():
    net = nn.Sequential()
    net.add(nn.Dense(1024, activation='relu'))
    # net.add(nn.Dropout(0.5))
    net.add(nn.Dense(512, activation='relu'))
    net.add(nn.Dense(1))
    net.initialize()
    return net

def log_rmse(net, features, labels):
    # to futher stabilize the value when the log is taken
    # set the value less than 1 as 1
    net_out = net(features)
    clipped_preds = np.clip(net_out, 1, float('inf'))
    return np.sqrt(2 * loss(np.log(clipped_preds), np.log(labels)).mean())

def train(net, train_features, train_labels, test_features, test_labels,
        num_epochs, learning_rate, weight_decay, batch_size):
    
    train_ls, test_ls = [], []
    train_iter = d2l_dx.load_array((train_features, train_labels), batch_size)
    trainer = gluon.Trainer(net.collect_params(), 'adam', 
                {'learning_rate': learning_rate, 'wd': weight_decay})
    for epoch in range(num_epochs):
        for X, y in train_iter:
            with autograd.record():
                l = loss(net(X), y)
            l.backward()
            trainer.step(batch_size)
        log_ls = log_rmse(net, train_features, train_labels)
        train_ls.append(log_ls)
        if test_labels is not None:
            test_ls.append(log_rmse(net, test_features, test_labels))
    return train_ls, test_ls

def get_k_fold_data(k, i, X, y):
    assert k > 1
    fold_size = X.shape[0] // k
    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        X_part, y_part = X[idx, :], y[idx]
        if j == i:
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = np.concatenate((X_train, X_part), axis=0)
            y_train = np.concatenate((y_train, y_part), axis=0)
    return X_train, y_train, X_valid, y_valid


def k_fold(k, X_train, y_train, num_epochs,
           learning_rate, weight_decay, batch_size):
    train_l_sum, valid_l_sum = 0, 0
    for i in range(k):
        data = get_k_fold_data(k, i, X_train, y_train)
        net = get_net()
        train_ls, valid_ls = train(net, *data, num_epochs, learning_rate,
                                   weight_decay, batch_size)
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]
        print('fold %d, train rmse: %f, valid rmse: %f' % (
            i, train_ls[-1], valid_ls[-1]))
    return train_l_sum / k, valid_l_sum / k

k, num_epochs, lr, weight_decay, batch_size = 5, 30, 0.5, 0.2, 128
train_l, valid_l = k_fold(k, train_features, train_labels, num_epochs, lr,
                          weight_decay, batch_size)

def train_and_pred(train_features, test_features, train_labels, test_data,
                    num_epochs, lr, weight_decay, batch_size):
    net = get_net()
    train_ls, _ = train(net, train_features, train_labels, None, None, num_epochs,
                    lr, weight_decay, batch_size)
    print('train rmse: {}'.format(train_ls[-1]))
    preds = net(test_features).asnumpy()
    test_data['SalePrice'] = pd.Series(preds.reshape(1, -1)[0])
    submission = pd.concat([test_data['Id'], test_data['SalePrice']], axis=1)
    submission.to_csv('submission.csv', index=False)

# train_and_pred(train_features, test_features, train_labels, test_data,
#                num_epochs, lr, weight_decay, batch_size)

