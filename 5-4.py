from mxnet import gluon, np, npx
from mxnet.gluon import nn
npx.set_np()

class CenteredLayer(nn.Block):
    def __init__(self, **kwargs):
        super(CenteredLayer, self).__init__(**kwargs)

    def forward(self, x):
        return x - x.mean()

layer = CenteredLayer()
print(layer(np.array([1,2,3,4,5])))

net = nn.Sequential()
net.add(nn.Dense(128), CenteredLayer())
net.initialize()
net.params
net.collect_params

gluon.ParameterDict
y = net(np.random.uniform(size=(4, 8)))

print(y.mean())
print(net.collect_params)
print(net.collect_params())

# params = gluon.ParameterDict()
# print(params.get('param2', shape=(2,3)))
# print(params)