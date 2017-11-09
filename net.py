import chainer
from chainer import links as L
from chainer import functions as F


class FX(chainer.Chain):
    def __init__(self, batch_size, input_size, output_size):
        super(FX, self).__init__()
        with self.init_scope():
            self.input_size = input_size
            self.output_size = output_size
            self.conv1 = L.Convolution2D(1, 5, (1, 100), stride=1)
            self.conv2 = L.Convolution2D(5, 10, (1, 50), stride=1)
            self.conv3 = L.Convolution2D(10, 20, (1, 10), stride=1)
            self.conv4 = L.Convolution2D(20, 40, (1, 3), stride=3)
            self.fc1 = L.Linear(None, 200)
            self.fc2 = L.Linear(None, 50)
            self.out = L.Linear(None, self.output_size)
            self.bnorm1 = L.BatchNormalization(5)
            self.bnorm2 = L.BatchNormalization(10)
            self.bnorm3 = L.BatchNormalization(20)
            self.batch_size = batch_size

    def __call__(self, x, t):
        h = self.predict(x)

        loss = F.softmax_cross_entropy(h, t)
        accuracy = F.accuracy(h, t)
        chainer.report({'loss': loss}, self)
        chainer.report({'accuracy': accuracy}, self)

        if chainer.config.train:
            return loss
        else:
            return h

    def predict(self, x):
        x = x.reshape(self.batch_size, 1, 1, self.input_size)
        h = F.relu(self.conv1(x))
        F.dropout(h, 0.2)
        h = F.relu(self.conv2(h))
        F.dropout(h, 0.2)
        h = F.relu(self.conv3(h))
        F.dropout(h, 0.2)
        h = F.relu(self.conv4(h))
        F.dropout(h, 0.2)
        h = self.fc1(h)
        h = self.fc2(h)
        h = self.out(h)
        return h
