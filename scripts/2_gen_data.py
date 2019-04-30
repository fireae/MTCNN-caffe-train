import sys
import numpy as np
import cv2
import numpy.random
from utils import IoU
import os


def make_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)


def gen_data(anno_file, std_size):
    save_dir = str(std_size)
    pos_save_dir = str(std_size) + "/pos"
    neg_save_dir = str(std_size) + "/neg"
    part_save_dir = str(std_size) + "/part"
    make_dir(save_dir)
    make_dir(pos_save_dir)
    make_dir(neg_save_dir)
    make_dir(part_save_dir)

    pos_file = open(os.path.join(save_dir, "pos_{}.txt".format(std_size)), "w")
    neg_file = open(os.path.join(save_dir, "neg_{}.txt".format(std_size)), "w")
    part_file = open(os.path.join(save_dir, "part_{}.txt".format(std_size)), "w")

    with open(anno_file, "r") as f:
        annotations = f.readlines()
    num = len(annotations)

    pos_idx = 0
    neg_idx = 0
    part_idx = 0
    idx = 0
    box_idx = 0

    for anno in annotations:
        anno = anno.strip().split(" ")
        img_path = anno[0]
        box = [float(anno[i]) for i in range(1, 5)]
        boxes = np.array(box, dtype=np.float32).reshape(-1, 4)
        image = cv2.imread(img_path)

        idx += 1
        if idx % 100 == 0:
            print("Processed {} images".format(idx))

        height, width, channel = image.shape

        neg_num = 0
        while neg_num < 100:
            size = np.random.randint(40, min(height, width) / 2)
            nx = np.random.randint(0, width - size)
            ny = np.random.randint(0, height - size)
            crop_box = np.array([nx, ny, nx + size, ny + size])
            iou = IoU(crop_box, boxes)
            if np.max(iou) < 0.3:
                cropped_im = image[ny : ny + size, nx : nx + size]
                resized_im = cv2.resize(
                    cropped_im, (std_size, std_size), interpolation=cv2.INTER_LINEAR
                )
                im_save_name = os.path.join(neg_save_dir, "{}.jpg".format(neg_idx))
                cv2.imwrite(im_save_name, resized_im)
                neg_file.write("{}/{}.jpg 0\n".format(neg_save_dir, neg_idx))
                neg_idx += 1
                neg_num += 1

        for box in boxes:
            x1, y1, x2, y2 = box
            w = x2 - x1 + 1
            h = y2 - y1 + 1

            if max(w, h) < 12 or x1 < 0 or y1 < 0 or w < 0 or h < 0:
                continue

            # generate pos and part samples
            for i in range(100):
                size = np.random.randint(
                    int(min(w, h) * 0.8), np.ceil(1.25 * max(w, h))
                )

                # delta here is the offset of box center
                delta_x = np.random.randint(-w * 0.2, w * 0.2)
                delta_y = np.random.randint(-h * 0.2, h * 0.2)

                nx1 = max(x1 + w / 2 + delta_x - size / 2, 0)
                ny1 = max(y1 + h / 2 + delta_y - size / 2, 0)
                nx2 = nx1 + size
                ny2 = ny1 + size

                if nx2 > width or ny2 > height:
                    continue
                crop_box = np.array([nx1, ny1, nx2, ny2])

                offset_x1 = (x1 - nx1) / float(size)
                offset_y1 = (y1 - ny1) / float(size)
                offset_x2 = (x2 - nx1) / float(size)
                offset_y2 = (y2 - ny1) / float(size)
                cropped_im = image[int(ny1) : int(ny2), int(nx1) : int(nx2), :]
                resized_im = cv2.resize(
                    cropped_im, (std_size, std_size), interpolation=cv2.INTER_LINEAR
                )
                box_ = box.reshape(1, -1)
                if IoU(crop_box, box_) >= 0.65:
                    save_file = os.path.join(pos_save_dir, "{}.jpg".format(pos_idx))
                    cv2.imwrite(save_file, resized_im)
                    s = "{}/{}.jpg 1 {} {} {} {}\n".format(
                        pos_save_dir,
                        pos_idx,
                        offset_x1,
                        offset_y1,
                        offset_x2,
                        offset_y2,
                    )
                    pos_file.write(s)
                    pos_idx += 1

                elif IoU(crop_box, box_) >= 0.4:
                    save_file = os.path.join(part_save_dir, "{}.jpg".format(part_idx))
                    cv2.imwrite(save_file, resized_im)
                    s = "{}/{}.jpg -1 {} {} {} {}\n".format(
                        part_save_dir,
                        part_idx,
                        offset_x1,
                        offset_y1,
                        offset_x2,
                        offset_y2,
                    )
                    part_file.write(s)
                    part_idx += 1
            box_idx += 1
            print(
                "Processed {} images, pos: {}, neg: {}, part: {}\n".format(
                    idx, pos_idx, neg_idx, part_idx
                )
            )


gen_data("2012_train.txt", 24)
gen_data("2012_train.txt", 48)
