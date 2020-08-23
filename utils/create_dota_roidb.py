import argparse
import os
import glob
import pickle
import tqdm

import cv2
import numpy as np


label_map = {
    "plane": 1,
    "ship": 2,
    "storage-tank": 3,
    "baseball-diamond": 4,
    "tennis-court": 5,
    "basketball-court": 6,
    "ground-track-field": 7,
    "harbor": 8,
    "bridge": 9,
    "large-vehicle": 10,
    "small-vehicle": 11,
    "helicopter": 12,
    "roundabout": 13,
    "soccer-ball-field": 14,
    "swimming-pool": 15
}

PATCH_SIZE = 1024
PATCH_STRIDE = 512


def parse_args():
    parser = argparse.ArgumentParser(description='Generate SimpleDet GroundTruth Database for the DOTA dataset')
    parser.add_argument('--data-dir', help='Path to the DOTA dataset, we assume a images and a labels_taskX_v1.0 folders', type=str)
    parser.add_argument('--split', type=str)

    args = parser.parse_args()
    return args.data_dir, args.split


def _process_single_label(label_path):
    with open(label_path) as fin:
        bbox_xyxy_list = []
        cls_list = []
        cls_name_list = []
        diff_list = []
        for line in fin:
            line = line.strip()
            if line[0] not in '0123456789':
                continue  # labels for task1 have header
            x1, y1, x2, y2, x3, y3, x4, y4, cls, diff = line.split(' ')
            x1, y1, x2, y2, x3, y3, x4, y4, diff = [float(_) for _ in [x1, y1, x2, y2, x3, y3, x4, y4, diff]]
            xmin = min([x1, x2, x3, x4])
            xmax = max([x1, x2, x3, x4])
            ymin = min([y1, y2, y3, y4])
            ymax = max([y1, y2, y3, y4])
            bbox_xyxy = [xmin, ymin, xmax, ymax]
            bbox_xyxy_list.append(bbox_xyxy)
            cls_list.append(label_map[cls])
            cls_name_list.append(cls)
            diff_list.append(diff)
        return np.array(bbox_xyxy_list), np.array(cls_list), cls_name_list, np.array(diff_list)


def _iou(box1, box2):
    x1, y1, x2, y2 = box1
    a1, b1, a2, b2 = box2
    if x2 <= a1 or a2 <= x1 or y2 <= b1 or b2 <= y1:
        return 0
    m1 = max(x1, a1)
    n1 = max(y1, b1)
    m2 = min(x2, a2)
    n2 = min(y2, b2)
    intersect = (n2 - n1) * (m2 - m1)
    assert intersect > 0
    union = (x2 - x1) * (y2 - y1) + (a2 - a1) * (b2 - b1) - intersect
    return intersect / union


def _ioa(box1, box2):
    x1, y1, x2, y2 = box1
    a1, b1, a2, b2 = box2
    if x2 <= a1 or a2 <= x1 or y2 <= b1 or b2 <= y1:
        return 0
    m1 = max(x1, a1)
    n1 = max(y1, b1)
    m2 = min(x2, a2)
    n2 = min(y2, b2)
    intersect = (n2 - n1) * (m2 - m1)
    assert intersect > 0
    area = max((x2 - x1) * (y2 - y1), (a2 - a1) * (b2 - b1))
    return intersect / area


def process_single_full_image_and_label(image_path, label_path, im_id):
    im = cv2.imread(image_path)
    h, w = im.shape[:2]

    bbox_xyxy_arr, cls_arr, cls_name_list, diff_arr = _process_single_label(label_path)

    roidb = []
    roirec = dict(
        gt_class=cls_arr,
        gt_class_name=cls_name_list,
        gt_bbox=bbox_xyxy_arr,
        gt_ignore=diff_arr,
        flipped=False,
        h=h,
        w=w,
        image_url=image_path,
        im_id=im_id,
    )
    roidb.append(roirec)
    return roidb


def process_single_image_and_label(image_path, label_path, image_patch_dir, im_id):
    im = cv2.imread(image_path)
    h, w = im.shape[:2]
    # x coords for grid points
    xcoords = set([0])
    j = PATCH_STRIDE
    while j + PATCH_SIZE < w:
        xcoords.add(j)
        j += PATCH_STRIDE
    if j not in xcoords and w - 1 - PATCH_SIZE > 0:
        xcoords.add(w - 1 - PATCH_SIZE)
    # y coords for grid points
    ycoords = set([0])
    i = PATCH_STRIDE
    while i + PATCH_SIZE < h:
        ycoords.add(i)
        i += PATCH_STRIDE
    if i not in ycoords and h - 1 - PATCH_SIZE > 0:
        ycoords.add(h - 1 - PATCH_SIZE)

    bbox_xyxy_arr, cls_arr, cls_name_list, diff_arr = _process_single_label(label_path)

    roidb = []
    image_base_name = os.path.basename(image_path)
    for y in sorted(ycoords):
        for x in sorted(xcoords):
            patch = im[y:y + PATCH_SIZE, x:x + PATCH_SIZE]
            image_patch_name = image_base_name.replace('.png', '_%d_%d.png' % (y, x))
            image_patch_path = os.path.join(image_patch_dir, image_patch_name)
            cv2.imwrite(image_patch_path, patch)

            keep_bbox = []
            keep_cls = []
            keep_cls_name = []
            keep_diff = []
            for i, box in enumerate(bbox_xyxy_arr):
                x1, y1, x2, y2 = box
                # box not in this region
                if x2 < x or x1 >= x + PATCH_SIZE or y2 < y or y1 >= y + PATCH_SIZE:
                    continue
                cx1, cx2 = np.clip([x1, x2], x, x + PATCH_SIZE)
                cy1, cy2 = np.clip([y1, y2], y, y + PATCH_SIZE)
                ioa = _ioa([cx1, cy1, cx2, cy2], [x1, y1, x2, y2])
                keep_bbox.append([cx1 - x, cy1 - y, cx2 - x, cy2 - y])
                keep_cls.append(cls_arr[i].item())
                keep_cls_name.append(cls_name_list[i])
                keep_diff.append(1 if ioa < 0.7 else diff_arr[i].item())

            if len(keep_bbox) == 0:
                keep_bbox = np.zeros((0, 4), dtype=np.float32)
                keep_cls = np.zeros((0, 1), dtype=np.float32)
                keep_diff = np.zeros((0, 1), dtype=np.float32)
            else:
                keep_bbox = np.array(keep_bbox, dtype=np.float32)
                keep_cls = np.array(keep_cls, dtype=np.float32)
                keep_diff = np.array(keep_diff, dtype=np.float32)

            roirec = dict(
                gt_class=keep_cls,
                gt_class_name=keep_cls_name,
                gt_bbox=keep_bbox,
                gt_ignore=keep_diff,
                flipped=False,
                h=PATCH_SIZE,
                w=PATCH_SIZE,
                offset_y=y,
                offset_x=x,
                image_url=image_patch_path,
                im_id=im_id,
            )
            roidb.append(roirec)
    return roidb


def create_roidb(data_dir, split, task='hbb'):
    # sanity check
    """
    |-- images
    |   |-- P0000.png
    |   |-- ...
    |   `-- P2805.png
    |-- labels_task2_v1.0
    |   |-- P0000.txt
    |   |-- ...
    |   `-- P2805.txt
    `-- labels_task2_v1.0
        |-- P0000.txt
        |-- ...
        `-- P2805.txt
    """
    assert os.path.exists(os.path.join(data_dir, 'images'))
    if task == 'obb':
        assert os.path.exists(os.path.join(data_dir, 'labels_task1_v1.0'))
        label_folder = 'labels_task1_v1.0'
    elif task == 'hbb':
        assert os.path.exists(os.path.join(data_dir, 'labels_task2_v1.0'))
        label_folder = 'labels_task2_v1.0'

    image_patch_dir = os.path.join(data_dir, 'image_patches')
    os.makedirs(image_patch_dir, exist_ok=True)
    image_paths = glob.glob(os.path.join(data_dir, 'images/*.png'))
    image_paths = sorted(image_paths)
    assert len(image_paths) > 0, 'image_paths is empty, can not find images in %s' % data_dir
    roidb = []
    for im_id, image_path in enumerate(tqdm.tqdm(image_paths)):
        label_path = image_path.replace('images', label_folder).replace('.png', '.txt')
        assert os.path.exists(label_path), '%s does not exist' % label_path
        if split == 'val_full':
            roidb += process_single_full_image_and_label(image_path, label_path, im_id)
        else:
            roidb += process_single_image_and_label(image_path, label_path, image_patch_dir, im_id)
    with open('data/cache/dota_v1_%s_%s.roidb' % (task, split), 'wb') as fout:
        pickle.dump(roidb, fout)


if __name__ == '__main__':
    data_dir, split = parse_args()
    os.makedirs("data/cache", exist_ok=True)
    create_roidb(data_dir, split, task='hbb')
