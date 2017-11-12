import chainer
import numpy as np
import random
import sys
import os.path
import datetime

class FxDataset:
    def __init__(self, train_file, input_size, output_size, time_length):
        self.input_size = input_size
        self.output_size = output_size
        self.time_length = time_length
        self.n = 11

        sys.stdout.write("loading data...")
        sys.stdout.flush()

        with open("diff_norm.txt") as diff_file:
            raw_diffs = diff_file.readlines()
        with open("rate.txt") as rate_file:
            raw_rates = rate_file.readlines()
        sys.stdout.write(" done.\n")
        sys.stdout.flush()

        sys.stdout.write("str to float...")
        sys.stdout.flush()
        self._diffs = [list(map(float, x.split(",")[1:self.n+1])) for x in raw_diffs]
        self._rates = [list(map(float, x.split(",")[1:self.n+1])) for x in raw_rates]
        self._time = [x.split(",")[0] for x in raw_diffs]
        sys.stdout.write(" done.\n")
        sys.stdout.flush()

    def __len__(self):
        """ データセットの数を返す関数 """
        return len(self._rates) - self.input_size - self.time_length - 1

    def split_random(self, n=None, m=None):
        """
        n: number of train datasets
        m: number of test datasets
        """
        print(len(self))
        if n is None:
            n = len(self)
        if m is None:
            m = n // 10
            n -= m
        if n + m > len(self):
            raise IndexError
        index = random.sample(range(len(self)), n + m)
        data = []
        print(len(self._rates))
        print(len(self._rates[0]))
        print("sampling...")
        for k in range(n + m):
            i = index[k]
            if self._rates[i + self.input_size][self.n-1] > self._rates[i + self.input_size + self.time_length][self.n-1]:
                up = 1
            else:
                up = 0
            data.append((np.array(self._diffs[i:i + self.input_size], dtype=np.float32),
                         np.array(up, dtype=np.int32)))
            if k % 1000 == 0:
                progress = k / (n + m)
                sys.stdout.write("\r[" + ("#" * int(progress * 40)) + (" " * (40 - int(progress * 40)) + "]"))
                sys.stdout.flush()
        sys.stdout.write("\r[" + ("#" * 40) + "]\n")
        sys.stdout.flush()

        print("done.")
        return data[:n], data[n:]
