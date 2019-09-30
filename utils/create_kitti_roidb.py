import argparse
import os
import glob
import pickle
import pprint
import json

import cv2
import numpy as np


label_map = dict(
    Car=1,
    Pedestrian=2,
    Cyclist=3
)


def parse_args():
    parser = argparse.ArgumentParser(description='Generate SimpleDet GroundTruth Database for the KITTI dataset')
    parser.add_argument('--data-dir', help='Path to KITTI dataset', type=str)

    args = parser.parse_args()
    return (args.data_dir, )

def get_calib_filename_from_frameno(frame_no, is_test=False):
    if is_test:
        return "testing/calib/%06d.txt" % frame_no
    else:
        return "training/calib/%06d.txt" % frame_no

def get_image_filename_from_frameno(frame_no, is_test=False):
    if is_test:
        return "testing/image_2/%06d.png" % frame_no
    else:
        return "training/image_2/%06d.png" % frame_no

def get_label_filename_from_frameno(frame_no, is_test=False):
    if is_test:
        return "testing/label_2/%06d.txt" % frame_no
    else:
        return "training/label_2/%06d.txt" % frame_no

def get_label(filename, debug=False):
    if isinstance(filename, int):
        filename = get_label_filename_from_frameno(filename)

    def parse_det_label(line):
        line = line.strip()
        parts = line.split(sep=" ")
        label_dict = dict(
            category=parts[0],
            truncated=float(parts[1]),
            occluded=int(parts[2]),
            observing_angle=float(parts[3]),
            bbox=[float(_) for _ in parts[4:8]],
            dimensions=[float(_) for _ in parts[8:11]],
            location=[float(_) for _ in parts[11:14]],
            heading=float(parts[14]))
        if debug:
            pprint.pprint(label_dict)
        return label_dict

    with open(filename, "r") as f:
        label_list = []
        for line in f:
            label_dict = parse_det_label(line)
            label_list.append(label_dict)
    return label_list

def create_roidb(data_dir):
    # sanity check
    """
        .
    |-- ImageSets
    |   |-- test.txt
    |   |-- train.txt
    |   |-- trainval.txt
    |   `-- val.txt
    |-- testing
    |   `-- image_2
    `-- training
        |-- calib
        |-- image_2
        `-- label_2
    """
    subdirs = [
        "ImageSets/test.txt",
        "ImageSets/train.txt",
        "ImageSets/trainval.txt",
        "ImageSets/val.txt",
        "testing/image_2",
        "training/calib",
        "training/image_2",
        "training/label_2"
    ]

    for subdir in subdirs:
        if not os.path.exists(os.path.join(data_dir, subdir)):
            raise Exception("{}/{} is not accessible".format(data_dir, subdir))

    # trainval sets
    for subset in ["train", "val", "trainval"]:
        all_roidb = []
        car_roidb = []
        ped_roidb = []
        cyc_roidb = []

        with open(os.path.join(data_dir, "ImageSets/{}.txt".format(subset))) as f:
            for im_id, line in enumerate(f):
                frameno = int(line.strip())
                image_filename = os.path.join(data_dir, get_image_filename_from_frameno(frameno))
                image_filename = os.path.abspath(image_filename)               
                h, w, _ = cv2.imread(image_filename, cv2.IMREAD_UNCHANGED).shape
                label_filename = os.path.join(data_dir, get_label_filename_from_frameno(frameno))
                label_list = get_label(label_filename)
                all_label, all_box = [], []
                car_label, car_box = [], []
                ped_label, ped_box = [], []
                cyc_label, cyc_box = [], []
                for label in label_list:
                    if label["category"] not in ["Car", "Pedestrian", "Cyclist"]:
                        continue
                    all_label.append(label_map[label["category"]])
                    all_box.append(label["bbox"])
                    if label["category"] == "Car":
                        car_label.append(1)
                        car_box.append(label["bbox"])
                    if label["category"] == "Pedestrian":
                        ped_label.append(1)
                        ped_box.append(label["bbox"])
                    if label["category"] == "Cyclist":
                        cyc_label.append(1)
                        cyc_box.append(label["bbox"])

                all_roidb.append(dict(
                    gt_class=np.array(all_label, dtype=np.float32),
                    gt_bbox=np.array(all_box, dtype=np.float32),
                    flipped=False,
                    h=h,
                    w=w,
                    image_url=image_filename,
                    im_id=im_id))
                car_roidb.append(dict(
                    gt_class=np.array(car_label, dtype=np.float32),
                    gt_bbox=np.array(car_box, dtype=np.float32),
                    flipped=False,
                    h=h,
                    w=w,
                    image_url=image_filename,
                    im_id=im_id))
                ped_roidb.append(dict(
                    gt_class=np.array(ped_label, dtype=np.float32),
                    gt_bbox=np.array(ped_box, dtype=np.float32),
                    flipped=False,
                    h=h,
                    w=w,
                    image_url=image_filename,
                    im_id=im_id))
                cyc_roidb.append(dict(
                    gt_class=np.array(cyc_label, dtype=np.float32),
                    gt_bbox=np.array(cyc_box, dtype=np.float32),
                    flipped=False,
                    h=h,
                    w=w,
                    image_url=image_filename,
                    im_id=im_id))

            with open("data/cache/kitti_{}_all.roidb".format(subset), "wb") as f:
                pickle.dump(all_roidb, f)
            with open("data/cache/kitti_{}_car.roidb".format(subset), "wb") as f:
                pickle.dump(car_roidb, f)
            with open("data/cache/kitti_{}_ped.roidb".format(subset), "wb") as f:
                pickle.dump(ped_roidb, f)
            with open("data/cache/kitti_{}_cyc.roidb".format(subset), "wb") as f:
                pickle.dump(cyc_roidb, f)
    
    # testing set
    subset = "test"
    roidb = []
    with open(os.path.join(data_dir, "ImageSets/{}.txt".format(subset))) as f:
        for im_id, line in enumerate(f):
            frameno = int(line.strip())
            image_filename = os.path.join(data_dir, get_image_filename_from_frameno(frameno, is_test=True))
            image_filename = os.path.abspath(image_filename)               
            h, w, _ = cv2.imread(image_filename, cv2.IMREAD_UNCHANGED).shape
            label, box = [], []
            roidb.append(dict(
                gt_class=np.array(label, dtype=np.float32),
                gt_bbox=np.array(box, dtype=np.float32),
                flipped=False,
                h=h,
                w=w,
                image_url=image_filename,
                im_id=im_id))
        with open("data/cache/kitti_{}.roidb".format(subset), "wb") as f:
            pickle.dump(roidb, f)


if __name__ == "__main__":
    os.makedirs("data/cache", exist_ok=True)
    create_roidb(*parse_args())
