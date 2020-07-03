import cv2
import numpy as np
import pickle as pkl
from detection_input import Stretch2DImageBbox


def visualize_original_input(input_record, no_display=False):
    image = input_record['image']
    gt_bbox = input_record['gt_bbox']
    for box in gt_bbox:
        cv2.rectangle(image, tuple(box[:2]), tuple(box[2:4]), color=(0, 255, 0))
    if no_display:
        from uuid import uuid4
        cv2.imwrite("debug_{}.jpg".format(uuid4()), image)
    else:
        cv2.imshow("imags", image)
        cv2.waitKey()


def prepare_input_list():
    with open('data/cache/coco_debug.roidb', 'rb') as fin:
        roidb = pkl.load(fin)
    input_list = []
    for roirec in roidb:
        input_record = {}
        input_record['image'] = cv2.imdecode(np.asarray(bytearray(roirec['image_bytes']), dtype=np.uint8), cv2.IMREAD_COLOR)
        input_record['gt_bbox'] = roirec['gt_bbox']
        input_list.append(input_record)
    return input_list


def test_stretch_2d_image_box():
    input_list = prepare_input_list()

    class StretchParam:
       ratio_range = (1 / 2, 2 / 1) 

    stretch = Stretch2DImageBbox(StretchParam)

    for input_record in input_list: 
        stretch.apply(input_record)
        visualize_original_input(input_record, no_display=True)


if __name__ == '__main__':
    test_stretch_2d_image_box()
