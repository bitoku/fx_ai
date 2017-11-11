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
    parser.add_argument('--dataset', '-d', type=str, default="USDJPY.txt.norm",
                        help='use dataset')
    parser.add_argument('--model', '-m', default='model_20',
                        help='path to the training model')
    parser.add_argument('--hours', '-p', type=int, default=24,
                        help='Predict hours')
    args = parser.parse_args()
    input_size = 60 * 24 * 1
    output_size = 60 * args.hours * 1
    model = FX(1, input_size, output_size)
    if args.gpu >= 0:
        model.to_gpu(chainer.cuda.get_device_from_id(args.gpu).use())
    serializers.load_npz(args.model, model)
    with open(args.input) as data:
        raw_rates = data.readlines()
        raw_rates.pop(0)
        rates = raw_rates[args.number:args.number + input_size + 1]
        rates = [float(x.split(",")[6]) for x in rates]
        diff = [(rates[i+1] - rates[i])*10 for i in range(len(rates) - 1)]
        diff_array = model.xp.array(diff, dtype=model.xp.float32)
        result = model.predict(diff_array)
        """
        for i in range(input_size):
            diff_array = model.xp.array(diff, dtype=model.xp.float32)
            result = model.predict(diff_array)
            p = model.xp.amax(result.data)
            print("predict:", p)
            diff.pop(0)
            diff.append(p)
        """
        pre_ans = rates[-1]
        rates = raw_rates[args.number: args.number + input_size + output_size + 1]
        rates = [float(x.split(",")[6]) for x in rates]
        post_ans = rates[-1]
        plt.plot(rates)
        for i, d in enumerate(result.data[0]):
            rates[i + input_size + 1] = rates[i + input_size] + d
        print(result.data[0])
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
