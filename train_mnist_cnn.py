import argparse

import matplotlib

matplotlib.use('Agg')
import chainer
from chainer import training
from chainer.training import extensions
from net import FX
from dataset import FxDataset

def main():
    parser = argparse.ArgumentParser(description='Chainer example: MNIST')
    parser.add_argument('--batchsize', '-b', type=int, default=1000,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=20,
                        help='Number of sweeps over the mini_cifar to train')
    parser.add_argument('--frequency', '-f', type=int, default=-1,
                        help='Frequency of taking a snapshot')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--dataset', '-d', default='rate_pair/USDJPY.txt',
                        help='File for train FxAi')
    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('')

    # Set up a neural network to train
    # Classifier reports softmax cross entropy loss and accuracy at every
    # iteration, which will be used by the PrintReport extension below.
    input_size = 24 * 60
    output_size = 2
    time_length = 60 * 24
    model = FX(args.batchsize, input_size, output_size)
    if args.gpu >= 0:
        # Make a specified GPU current
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()  # Copy the model to the GPU

    # Setup an optimizer
    optimizer = chainer.optimizers.AdaDelta()
    optimizer.setup(model)

    # Load the MNIST mini_cifar
    datasets = FxDataset(args.dataset, input_size, output_size, time_length)
    train, test = datasets.split_random(100000, 10000)

    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test, args.batchsize,
                                                 repeat=False, shuffle=False)

    # Set up a trainer
    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    # Evaluate the model with the test mini_cifar for each epoch
    trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu))

    trainer.extend(extensions.dump_graph('main/loss'))

    frequency = args.epoch if args.frequency == -1 else max(1, args.frequency)
    trainer.extend(extensions.snapshot(), trigger=(frequency, 'epoch'))
    trainer.extend(extensions.snapshot_object(model, 'model_{.updater.epoch}'),
                   trigger=(frequency, 'epoch'))

    trainer.extend(extensions.LogReport())
    if extensions.PlotReport.available():
        trainer.extend(
            extensions.PlotReport(['main/loss', 'validation/main/loss', 'main/accuracy', 'validation/main/accuracy'],
                                  'epoch', file_name='loss.png'))

    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss', 'main/accuracy', 'validation/main/accuracy',
         'elapsed_time']))

    trainer.extend(extensions.ProgressBar())

    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)

    trainer.run()


if __name__ == '__main__':
    main()
