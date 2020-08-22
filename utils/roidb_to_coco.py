import numpy as np
import json
import tempfile

from pycocotools.coco import COCO
from operator_py.detectron_bbox_utils import xyxy_to_xywh


def roidb_to_coco_ex(roidb):
    # The whole coco dataset
    dataset = {
        'categories': [],
        'images': [],
        'annotations': []
    }

    category_ids = {}
    obj_id = 0
    for roirec in roidb:
        dataset['images'].append({
            'image_path': roirec['image_url'],
            'id': roirec['im_id'],
            'width': roirec['w'],
            'height': roirec['h'],
        })
        if len(roirec['gt_bbox']) > 0:
            roirec['gt_bbox'] = xyxy_to_xywh(roirec['gt_bbox'])
        if 'gt_ignore' not in roirec:
            roirec['gt_ignore'] = np.zeros_like(roirec['gt_class'])
        if 'gt_class_name' not in roirec:
            roirec['gt_class_name'] = [None] * len(roirec['gt_class'])
        for bbox, cls, cls_name, ignore in zip(roirec['gt_bbox'], roirec['gt_class'], roirec['gt_class_name'], roirec['gt_ignore']):
            x, y, h, w = bbox.tolist()
            dataset["annotations"].append({
                'area': h * w,
                'bbox': [x, y, h, w],
                'category_id': int(cls),
                'id': obj_id,
                'image_id': roirec['im_id'],
                'iscrowd': 0,
                'ignore': int(ignore),
            })
            obj_id += 1
            category_ids[int(cls)] = cls_name or int(cls)
    for class_id in category_ids:
        dataset['categories'].append({
            'id': class_id,
            'name': category_ids[class_id],
            'supercategory': 'none'
        })

    with tempfile.NamedTemporaryFile(mode="w") as f:
        json.dump(dataset, f)
        f.flush()
        coco = COCO(f.name)

    return coco, dataset


def roidb_to_coco(roidb):
    coco, _ = roidb_to_coco_ex(roidb)
    return coco

