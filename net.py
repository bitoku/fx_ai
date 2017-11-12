import chainer
from chainer import links as L
from chainer import functions as F


class FX(chainer.Chain):
    def __init__(self, batch_size, input_size, output_size):
        super(FX, self).__init__()
        with self.init_scope():
            self.input_size = input_size
            self.output_size = output_size
            self.conv1 = L.Convolution2D(1, 64, (11, 30), stride=5, pad=(0, 10))
            self.conv2 = L.Convolution2D(64, 128, (1, 15), stride=3)
            self.conv3 = L.Convolution2D(128, 256, (1, 10), stride=2)
            self.conv4 = L.Convolution2D(256, 256, (1, 3), stride=1)
            self.fc1 = L.Linear(None, 2000)
            self.fc2 = L.Linear(None, 1000)
            self.out = L.Linear(None, self.output_size)
            self.bnorm1 = L.BatchNormalization(64)
            self.bnorm2 = L.BatchNormalization(128)
            self.bnorm3 = L.BatchNormalization(256)
            self.bnorm4 = L.BatchNormalization(256)
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
        x = x.reshape(self.batch_size, 1, 11, self.input_size)
        h = F.relu(self.bnorm1(self.conv1(x)))
        F.dropout(h, 0.2)
        h = F.relu(self.bnorm2(self.conv2(h)))
        h = F.average_pooling_2d(h, (1,2), stride=1)
        F.dropout(h, 0.2)
        h = F.relu(self.bnorm3(self.conv3(h)))
        F.dropout(h, 0.2)
        h = F.relu(self.bnorm4(self.conv4(h)))
        h = F.average_pooling_2d(h, (1,2), stride=1)
        F.dropout(h, 0.2)
        h = self.fc1(h)
        h = self.fc2(h)
        h = self.out(h)
        return h
