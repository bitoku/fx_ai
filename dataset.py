import chainer
import numpy as np
import random
import sys
import os.path

class FxDataset:
    def __init__(self, train_file, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        if not os.path.exists(train_file + ".norm"):
            sys.stdout.write("loading data...")
            sys.stdout.flush()
            with open(train_file) as file:
                raw_rates = file.readlines()
                self.first_date = raw_rates[0].split(",")[1]
                self.first_time = raw_rates[0].split(",")[2]
                raw_rates.pop(0)
                rates = [float(x.split(",")[6]) for x in raw_rates]
                sys.stdout.write(" done.\n")

                sys.stdout.write("normalizing data...")
                sys.stdout.flush()
                self._diff = [(rates[i+1] - rates[i])*10 for i in range(len(rates)-1)]
                sys.stdout.write(" done.\n")
                sys.stdout.flush()

            with open(train_file + ".norm", "w") as file:
                sys.stdout.write("save normalized data...")
                sys.stdout.flush()
                file.write("{0},{1}".format(self.first_date, self.first_time))
                for x in self._diff:
                    file.write("\n{0}".format(x))
                sys.stdout.write(" done.\n")
                sys.stdout.flush()
        else:
            with open(train_file + ".norm") as file:
                raw_diff = file.readlines()
                self.first_date = raw_diff[0].split(",")[0]
                self.first_time = raw_diff[0].split(",")[1]
                raw_diff.pop(0)
                self._diff = [float(x) for x in raw_diff]


    def __len__(self):
        """ データセットの数を返す関数 """
        return len(self._diff) - self.input_size - self.output_size - 1

    def split_random(self, n=None, m=None):
        """
        n: number of train datasets
        m: number of test datasets
        """
        if n is None:
            n = len(self)
        if m is None:
            m = n // 10
            n -= m
        if n + m > len(self):
            raise IndexError
        index = random.sample(range(len(self)), n + m)
        data = []
        print("sampling...")
        for k in range(n + m):
            i = index[k]
            data.append((np.array(self._diff[i:i + self.input_size], dtype=np.float32),
                         np.array(self._diff[i + self.input_size:i + self.input_size + self.output_size],
                                  dtype=np.float32)))
            if k % 1000 == 0:
                progress = k / (n + m)
                sys.stdout.write("\r[" + ("#" * int(progress * 40)) + (" " * (40 - int(progress * 40)) + "]"))
                sys.stdout.flush()
        sys.stdout.write("\r[" + ("#" * 40) + "]\n")
        sys.stdout.flush()

        print("done.")
        return data[:n], data[n:]
