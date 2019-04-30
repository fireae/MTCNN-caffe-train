import numpy as np
import cv2
import h5py
import sys
import random


def make_mtcnn_h5(label_file, h5_path, std_size):
    f = open(label_file, "r")
    lines = f.readlines()
    random.shuffle(lines)
    random.shuffle(lines)
    img_array = []
    labels = []
    boxes = []
    for line in lines:
        line = line.strip().split(" ")
        im_name = line[0]
        label = int(line[1])
        if len(line) < 3:
            box = [0, 0, 0, 0]
        else:
            box = [float(line[i]) for i in range(2, 6)]
        img = cv2.imread(im_name)
        img = cv2.resize(img, (std_size, std_size))
        img_forward = np.array(img, dtype=np.float32)
        img_forward = np.transpose(img_forward, (2, 0, 1))
        img_forward = (img_forward - 127.5) * 0.0078125
        img_array.append(img_forward)
        labels.append(label)
        boxes.append(box)
    img_array = np.array(img_array, dtype=np.float32)
    label = np.array(labels, dtype=np.int32).reshape(-1, 1)
    print(label.shape)
    boxes = np.array(boxes, dtype=np.float32)
    print(boxes.shape)
    with h5py.File(h5_path, "w") as h5f:
        h5f["data"] = img_array
        h5f["label"] = labels
        h5f["regression"] = boxes


make_mtcnn_h5("12_all.txt", "12_all.h5", 12)
make_mtcnn_h5("24_all.txt", "24_all.h5", 24)
make_mtcnn_h5("48_all.txt", "48_all.h5", 48)

