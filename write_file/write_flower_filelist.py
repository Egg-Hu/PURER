import numpy as np
from os import listdir
from os.path import isfile, isdir, join
import os
import json
import random
import csv

import scipy.io

data_path = './DFL2Ldata/flower/jpg'
savedir = '/./DFL2Ldata/flower/split/'
label_path='./DFL2Ldata/flower/imagelabels.mat'
labels_mat = scipy.io.loadmat(label_path)
os.makedirs(savedir, exist_ok=True)
split_list = ['meta_train', 'meta_val', 'meta_test']
SPLITS = {
    'meta_train': [90, 38, 80, 30, 29, 12, 43, 27, 4, 64, 31, 99, 8, 67, 95, 77,
              78, 61, 88, 74, 55, 32, 21, 13, 79, 70, 51, 69, 14, 60, 11, 39,
              63, 37, 36, 28, 48, 7, 93, 2, 18, 24, 6, 3, 44, 76, 75, 72, 52,
              84, 73, 34, 54, 66, 59, 50, 91, 68, 100, 71, 81, 101, 92, 22,
              33, 87, 1, 49, 20, 25, 58],
    'meta_val': [10, 16, 17, 23, 26, 47, 53, 56, 57, 62, 82, 83, 86, 97, 102],
    'meta_test': [5, 9, 15, 19, 35, 40, 41, 42, 45, 46, 65, 85, 89, 94, 96, 98],
    'all': list(range(1, 103)),
}
print(len(SPLITS['meta_train']))
print(len(SPLITS['meta_val']))
print(len(SPLITS['meta_test']))
for mode in split_list:
    num = 0
    file_list = []
    label_list = []
    split = SPLITS[mode]
    for idx, label in enumerate(labels_mat['labels'][0], start=1):
        if label in split:
            file_list.append( os.path.join(data_path,'image_%05d.jpg'%(idx)))
            label_list.append(label)
            num=num+1
    print('split_num:', num)
    fo = open(savedir + mode + ".csv", "w", newline='')
    writer = csv.writer(fo)
    writer.writerow(['filename', 'label'])
    temp = np.array(list(zip(file_list, label_list)))
    writer.writerows(temp)
    fo.close()
    print("%s -OK" % mode)
