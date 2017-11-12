# coding: utf-8
import sys
import glob
import datetime
import numpy as np

"""
with open("rate.txt") as f:
    rate = f.readlines()
    print(len(rate))
    """

rates = {}
for file in sys.argv[1:]:
    with open(file) as f:
        raw_rates = f.readlines()
        raw_rates = raw_rates[-1000000:]
        for raw_rate in raw_rates:
            time = raw_rate.split(",")[1] + raw_rate.split(",")[2]
            if time in rates:
                rates[time].append(float(raw_rate.split(",")[6]))
            else:
                rates[time] = [float(raw_rate.split(",")[6])]
        print(file)

delete_key = []
for key, val in rates.items():
    if len(val) < 11:
        delete_key.append(key)
for key in delete_key:
    del rates[key]
rates = sorted(rates.items())

n = 11
rates_list = [[rates[i][0] for i in range(len(rates))]]
for j in range(n):
    rates_list.append([rates[i][1][j] for i in range(len(rates))])
print(rates_list[0][0], rates_list[1][0])

diff_list = [rates_list[0][:-2]]
for j in range(1, n+1):
    diff = [rates_list[j][i+1] - rates_list[j][i] for i in range(len(rates)-1)]
    diff_list.append(diff)
print(diff_list[0][0], diff_list[1][0])

diff_norm_list = [diff_list[0]]
for j in range(1, n+1):
    std = np.std(diff_list[j])
    avg = np.average(diff_list[j])
    diff_norm_list.append([(diff_list[j][i] - avg) / std for i in range(len(diff_list[j]))])
print(diff_norm_list[0][0], diff_norm_list[1][0])

with open("diff_norm.txt", "w") as f:
    for i in range(len(diff_norm_list[0])):
        f.write("{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11}\n".format(diff_norm_list[0][i], diff_norm_list[1][i],
                                                                             diff_norm_list[2][i], diff_norm_list[3][i],
                                                                             diff_norm_list[4][i], diff_norm_list[5][i],
                                                                             diff_norm_list[6][i], diff_norm_list[7][i],
                                                                             diff_norm_list[8][i], diff_norm_list[9][i],
                                                                             diff_norm_list[10][i],
                                                                             diff_norm_list[11][i]))
