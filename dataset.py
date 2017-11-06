import chainer
import numpy as np


class FxDataset(chainer.dataset.DatasetMixin):
    def __init__(self, train_file):
        self.unit_length = 24 * 60
        with open(train_file) as file:
            raw_rates = file.readlines()
            raw_rates.pop(0)
            self._rates = [x.split(",")[6] for x in raw_rates]
            self._first_date = raw_rates[0].split(",")[1]
            self._first_time = raw_rates[0].split(",")[1]

    def __len__(self):
        """ データセットの数を返す関数 """
        return len(self._paths) - self.unit_length - 1
