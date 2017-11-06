#!/usr/bin/env python
try:
    import matplotlib
    matplotlib.use('Agg')
except ImportError:
    pass
import argparse

import chainer
from PIL import Image
from chainer import serializers

from net import FX


def main():
    parser = argparse.ArgumentParser(description='Chainer Fx')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--input', '-i', type=str, default="USDJPY.txt",
                        help='use input_file')
    parser.add_argument('--dataset', '-d', type=str, default="USDJPY.txt",
                        help='use dataset')
    parser.add_argument('--model', '-m', default='model_50',
                        help='path to the training model')
    args = parser.parse_args()
    model = FX()
    if args.gpu >= 0:
        model.to_gpu(chainer.cuda.get_device_from_id(args.gpu).use())
    serializers.load_npz(args.model, model)
    with open(args.data) as data:
        data
        result = model.predict(img_array)
        print("predict:", model.xp.argmax(result.data))
    '''
    try:
        img = Image.open(args.image).convert("L").resize((28,28))
    except :
        print("invalid input")
        return
    '''

if __name__ == '__main__':
    main()
