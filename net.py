import chainer
from chainer import links as L
from chainer import functions as F

class MLP(chainer.Chain):
    def __init__(self, n_units, n_out):
        super(MLP, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(None, n_units)  # n_in -> n_units
            self.l2 = L.Linear(None, n_units)  # n_units -> n_units
            self.l3 = L.Linear(None, n_out)  # n_units -> n_out

    def __call__(self, x, t=None):
        h = F.relu(self.l1(x))
        h = F.relu(self.l2(h))
        h = self.l3(h)

        loss = F.softmax_cross_entropy(h, t)
        accuracy = F.accuracy(h, t)
        chainer.report({'loss': loss}, self)
        chainer.report({'accuracy': accuracy}, self)

        if chainer.config.train:
            return loss
        else:
            return h

    def predict(self, x):
        h = F.relu(self.l1(x))
        h = F.relu(self.l2(h))
        h = self.l3(h)
        return h


class MNIST_CNN(chainer.Chain):
    def __init__(self, batch_size):
        super(MNIST_CNN, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(1, 5, (1, 3))
            self.conv2 = L.Convolution2D(5, 10, (1, 3))
            self.conv3 = L.Convolution2D(10, 20, (1, 3))
            self.l_out = L.Linear(None, 1)
            self.bnorm1 = L.BatchNormalization(5)
            self.bnorm2 = L.BatchNormalization(10)
            self.bnorm3 = L.BatchNormalization(20)
            self.batch_size = batch_size

    def __call__(self, x, t):
        # print("x.shape:", x.shape)
        x = x.reshape(self.batch_size, 1, 1, 1440)
        # print("x.reshape:", x.shape)
        h = F.relu(self.conv1(x))
        #h = self.bnorm1(h)
        # print("x.reshape:", x.shape)
        # print("h.shape:", h.shape)
        # h = h.reshape(1, 5, 14, 14))
        # print("h.reshape:", h.shape)
        h = F.relu(self.conv2(h))
        #h = self.bnorm2(h)
        # print("h.shape2:", h.shape)
        h = F.relu(self.conv3(h))
        #h = self.bnorm3(h)
        h = self.l_out(h)
        # print("h.shape3:", h.shape)

        loss = F.mean_squared_error(h, t)
        # accuracy = F.accuracy(h, t)
        chainer.report({'loss': loss}, self)
        # chainer.report({'accuracy': accuracy}, self)

        if chainer.config.train:
            return loss
        else:
            return h


class Cifar_CNN(chainer.Chain):
    def __init__(self, n_out):
        super(Cifar_CNN, self).__init__()
        with self.init_scope():
            self.conv1_1 = L.Convolution2D(3, 32, 3)
            self.l1 = L.Linear(None, 512)
            self.l_out = L.Linear(512, n_out)

    def __call__(self, x, t):
        h = F.max_pooling_2d(F.relu(self.conv1_1(x)), 2)
        h = F.relu(self.l1(h))
        h = self.l_out(h)

        t = self.xp.asarray(t, self.xp.int32)
        loss = F.softmax_cross_entropy(h, t)
        accuracy = F.accuracy(h, t)
        chainer.report({'loss': loss}, self)
        chainer.report({'accuracy': accuracy}, self)

        if chainer.config.train:
            return loss
        else:
            return h

    def predict(self, x):
        h = F.max_pooling_2d(F.relu(self.conv1_1(x)), 2)
        h = F.relu(self.l1(h))
        h = self.l_out(h)
        predicts = F.argmax(h, axis=1)
        return predicts.data
