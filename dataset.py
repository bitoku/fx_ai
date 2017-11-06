import chainer
import numpy as np
import random

class FxDataset:
    def __init__(self, train_file):
        self.unit_length = 24 * 60
        with open(train_file) as file:
            raw_rates = file.readlines()
            raw_rates.pop(0)
            self._rates = [x.split(",")[6] for x in raw_rates]
            self._first_date = raw_rates[0].split(",")[1]
            self._first_time = raw_rates[0].split(",")[1]
        print("data was loaded.")

    def __len__(self):
        """ データセットの数を返す関数 """
        return len(self._rates) - self.unit_length - 1

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
        for i in index:
            data.append((np.array(self._rates[i:i + self.unit_length], dtype=np.float32),
                                  np.array(self._rates[i + self.unit_length + 1], dtype=np.float32)))
        print("done.")
        return data[:n], data[n:]
