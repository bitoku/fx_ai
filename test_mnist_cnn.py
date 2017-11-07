#!/usr/bin/env python
try:
    import matplotlib
    matplotlib.use('Agg')
except ImportError:
    pass
import matplotlib.pyplot as plt
import argparse

import chainer
from PIL import Image
from chainer import serializers

from net import FX
from dataset import FxDataset


def main():
    parser = argparse.ArgumentParser(description='Chainer Fx')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--input', '-i', type=str, default="USDJPY.txt",
                        help='use input_file')
    parser.add_argument('--number', '-n', type=int, default=1,
                        help='use number of line')
    parser.add_argument('--dataset', '-d', type=str, default="USDJPY.txt",
                        help='use dataset')
    parser.add_argument('--model', '-m', default='model_50',
                        help='path to the training model')
    args = parser.parse_args()
    model = FX(1)
    if args.gpu >= 0:
        model.to_gpu(chainer.cuda.get_device_from_id(args.gpu).use())
    serializers.load_npz(args.model, model)
    with open(args.input) as data:
        raw_rates = data.readlines()
        raw_rates.pop(0)
        rates = raw_rates[args.number:args.number + 1440]
        rates = [float(x.split(",")[6]) for x in rates]
        rates_array = model.xp.array(rates, dtype=model.xp.float32)
        result = model.predict(rates_array)
        """
        for i in range(60 * 24):
            rates_array = model.xp.array(rates, dtype=model.xp.float32)
            result = model.predict(rates_array)
            p = model.xp.amax(result.data)
            print("predict:", p, raw_rates[args.number + 1440 + i].split(",")[6])
            rates.pop(0)
            rates.append(p)
        """
        print(result.data[0])
        plt.plot(result.data[0])
        rates = raw_rates[args.number + 1440: args.number + 2880]
        rates = [float(x.split(",")[6]) for x in rates]
        plt.plot(rates)
        plt.savefig("predict.png")
    '''
    try:
        img = Image.open(args.image).convert("L").resize((28,28))
    except :
        print("invalid input")
        return
    '''

if __name__ == '__main__':
    main()
