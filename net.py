import chainer
from chainer import links as L
from chainer import functions as F


class FX(chainer.Chain):
    def __init__(self, batch_size, input_size, output_size):
        super(FX, self).__init__()
        with self.init_scope():
            self.batch_size = batch_size
            self.input_size = input_size
            self.output_size = output_size
            self.conv1 = L.Convolution2D(1, 5, (1, 100), stride=1)
            self.conv2 = L.Convolution2D(5, 10, (1, 50), stride=1)
            self.conv3 = L.Convolution2D(10, 20, (1, 10), stride=1)
            self.conv4 = L.Convolution2D(20, 30, (1, 3), stride=3)
            self.fc1 = L.Linear(None, 100)
            self.fc2 = L.Linear(None, self.output_size)
            self.bnorm1 = L.BatchNormalization(5)
            self.bnorm2 = L.BatchNormalization(10)
            self.bnorm3 = L.BatchNormalization(20)


    def __call__(self, x, t):
        h = self.predict(x)

        loss = F.mean_squared_error(h, t)
        # accuracy = F.accuracy(h, t)
        chainer.report({'loss': loss}, self)
        # chainer.report({'accuracy': accuracy}, self)

        if chainer.config.train:
            return loss
        else:
            return h

    def predict(self, x):
        # print("x.shape:", x.shape)
        x = x.reshape(self.batch_size, 1, 1, self.input_size)
        #print("x.reshape:", x.shape)
        #h = F.average_pooling_2d(F.relu(self.conv1(x)), (1, 3))
        h = F.relu(self.conv1(x))
        #h = self.bnorm1(h)
        #print("h.shape1:", h.shape)
        h = F.average_pooling_2d(F.relu(self.conv2(h)), (1, 2))
        #h = F.relu(self.conv2(h))
        #h = self.bnorm2(h)
        #print("h.shape2:", h.shape)
        h = F.relu(self.conv3(h))
        #h = self.bnorm3(h)
        #print("h.shape3:", h.shape)
        #h = F.relu(self.conv4(h))
        #print("h.shape4:", h.shape)
        #h = self.fc1(h)
        h = self.fc2(h)
        return h
