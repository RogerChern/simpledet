import argparse
import importlib
import os
import time

from utils.load_model import load_checkpoint
from utils.patch_config import patch_config_as_nothrow
from core.detection_module import DetModule

import cv2
import mxnet as mx
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description='Test detector inference speed')
    # general
    parser.add_argument('--config', help='config file path', type=str, required=True)
    parser.add_argument('--path', help='specify input 2d image path', type=str, required=True)
    parser.add_argument('--gpu', help='GPU index', type=int, default=0)
    parser.add_argument('--ckpt', help='checkpoint path', type=str)
    args = parser.parse_args()

    config = importlib.import_module(args.config.replace('.py', '').replace('/', '.'))
    args.config = config
    return args


if __name__ == "__main__":
    os.environ["MXNET_CUDNN_AUTOTUNE_DEFAULT"] = os.environ.get("MXNET_CUDNN_AUTOTUNE_DEFAULT", "0")

    args = parse_args()

    pGen, pKv, pRpn, pRoi, pBbox, pDataset, pModel, pOpt, pTest, \
        transform, data_name, label_name, metric_list = args.config.get_config(is_train=False)
    pGen = patch_config_as_nothrow(pGen)
    pKv = patch_config_as_nothrow(pKv)
    pRpn = patch_config_as_nothrow(pRpn)
    pRoi = patch_config_as_nothrow(pRoi)
    pBbox = patch_config_as_nothrow(pBbox)
    pDataset = patch_config_as_nothrow(pDataset)
    pModel = patch_config_as_nothrow(pModel)
    pOpt = patch_config_as_nothrow(pOpt)
    pTest = patch_config_as_nothrow(pTest)

    sym = pModel.test_symbol

    # load dataset
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    from utils.roidb_to_coco import roidb_to_coco
    if pTest.coco.annotation is not None:
        coco = COCO(pTest.coco.annotation)
    else:
        coco = roidb_to_coco(roidbs_all)

    # load an image and convert to a data batch
    input_record = {"image_url": args.path, "gt_bbox": np.zeros(shape=(0, 4), dtype=np.float32), "gt_class": np.zeros(shape=(0, 1), dtype=np.float32)}
    for trans in transform:
        trans.apply(input_record)
    im_id = mx.nd.array([1])
    rec_id = mx.nd.array([1])
    input_record["data"] = input_record["data"][None, :]
    input_record["im_info"] = input_record["im_info"][None, :]
    data = [mx.nd.array(x) for x in [input_record["data"], input_record["im_info"], im_id, rec_id]]

    data_names = ["data", "im_info", "im_id", "rec_id"]
    data_shape = [[1, 3, 1333, 1333], [1, 3], [1], [1]]
    data_shape = [(name, shape) for name, shape in zip(data_names, data_shape)]
    data_batch = mx.io.DataBatch(data=data)

    # load params
    arg_params, aux_params = load_checkpoint(pTest.model.prefix, pTest.model.epoch)
    if pModel.process_weight is not None:
        pModel.process_weight(sym, arg_params, aux_params)

    ctx = mx.gpu(args.gpu)
    mod = DetModule(sym, data_names=data_names, context=ctx)
    mod.bind(data_shapes=data_shape, for_training=False)
    mod.set_params(arg_params, aux_params)

    mod.forward(data_batch, is_train=False)
    rid, id, info, cls, box = mod.get_outputs()
    rid, id, info, cls, box = rid.squeeze().asnumpy(), id.squeeze().asnumpy(), info.squeeze().asnumpy(), cls.squeeze().asnumpy(), box.squeeze().asnumpy()

    scale = info[2]  # h_raw, w_raw, scale
    box = box / scale  # scale to original image scale
    cls_score = cls[:, 1:]   # remove background
    # TODO: the output shape of class_agnostic box is [n, 4], while class_aware box is [n, 4 * (1 + class)]
    bbox_xyxy = box[:, 4:] if box.shape[1] != 4 else box


    # do nms
    if callable(pTest.nms.type):
        nms = pTest.nms.type(pTest.nms.thr)
    else:
        from operator_py.nms import py_nms_wrapper
        nms = py_nms_wrapper(pTest.nms.thr)

    final_dets = {}
    for cid in range(cls_score.shape[1]):
        score = cls_score[:, cid]
        if bbox_xyxy.shape[1] != 4:
            cls_box = bbox_xyxy[:, cid * 4:(cid + 1) * 4]
        else:
            cls_box = bbox_xyxy
        valid_inds = np.where(score > pTest.min_det_score)[0]
        box = cls_box[valid_inds]
        score = score[valid_inds]
        box_xyxys = np.concatenate([box, score[:, None]], axis=1).astype(np.float32)
        box_xyxys = nms(box_xyxys)
        dataset_cid = coco.getCatIds()[cid]
        final_dets[dataset_cid] = box_xyxys

    result = []
    for cid in final_dets:
        det = final_dets[cid]
        if det.shape[0] == 0:
            continue
        scores = det[:, -1]
        xs = det[:, 0]
        ys = det[:, 1]
        ws = det[:, 2] - xs + 1
        hs = det[:, 3] - ys + 1
        result += [
            {'category_id': int(cid),
             'bbox': [float(xs[k]), float(ys[k]), float(ws[k]), float(hs[k])],
             'score': float(scores[k])}
            for k in range(det.shape[0])
        ]
    result = sorted(result, key=lambda x: x['score'])[-pTest.max_det_per_image:]
    for i, x in enumerate(result, start=1):
        if x['score'] > 0.5:
            print("deteciton #%d" % i)
            print("category: %s" % coco.cats[x['category_id']]['name'])
            print("bbox: x=%.1f, y=%.1f, w=%.1f, h=%.1f" % tuple(x['bbox']))
            print("score: %.3f\n" % x['score'])

