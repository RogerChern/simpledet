import logging
import numpy as np


def summarize(coco_eval, ap=1, iouThr=None, areaRng='all', maxDets=100):
    p = coco_eval.params
    iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'
    titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
    typeStr = '(AP)' if ap == 1 else '(AR)'
    iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
        if iouThr is None else '{:0.2f}'.format(iouThr)

    aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
    mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
    if ap == 1:
        # dimension of precision: [TxRxKxAxM]
        s = coco_eval.eval['precision']
        # IoU
        if iouThr is not None:
            t = np.where(iouThr == p.iouThrs)[0]
            s = s[t]
        s = s[:,:,:,aind,mind]
    else:
        # dimension of recall: [TxKxAxM]
        s = coco_eval.eval['recall']
        if iouThr is not None:
            t = np.where(iouThr == p.iouThrs)[0]
            s = s[t]
        s = s[:,:,aind,mind]
    if len(s[s > -1]) == 0:
        mean_s = -1
    else:
        mean_s = np.mean(s[s > -1])
    print(iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s))
    return mean_s


def official_summary(coco_eval):
    print("Evaluate with the offical coco metric")
    stats = [0] * 12
    stats[0] = summarize(coco_eval, 1)
    stats[1] = summarize(coco_eval, 1, iouThr=.5, maxDets=coco_eval.params.maxDets[2])
    stats[2] = summarize(coco_eval, 1, iouThr=.75, maxDets=coco_eval.params.maxDets[2])
    stats[3] = summarize(coco_eval, 1, areaRng='small', maxDets=coco_eval.params.maxDets[2])
    stats[4] = summarize(coco_eval, 1, areaRng='medium', maxDets=coco_eval.params.maxDets[2])
    stats[5] = summarize(coco_eval, 1, areaRng='large', maxDets=coco_eval.params.maxDets[2])
    stats[6] = summarize(coco_eval, 0, maxDets=coco_eval.params.maxDets[0])
    stats[7] = summarize(coco_eval, 0, maxDets=coco_eval.params.maxDets[1])
    stats[8] = summarize(coco_eval, 0, maxDets=coco_eval.params.maxDets[2])
    stats[9] = summarize(coco_eval, 0, areaRng='small', maxDets=coco_eval.params.maxDets[2])
    stats[10] = summarize(coco_eval, 0, areaRng='medium', maxDets=coco_eval.params.maxDets[2])
    stats[11] = summarize(coco_eval, 0, areaRng='large', maxDets=coco_eval.params.maxDets[2])
    print("Easy copy AP: %s" % ", ".join(str("%2.1f" % (_ * 100)) for _ in stats[:1]))
    print("Easy copy IoU=0.5, 0.75: %s" % ", ".join(str("%2.1f" % (_ * 100)) for _ in stats[1:3]))
    print("Easy copy s, m, l: %s" % ", ".join(str("%2.1f" % (_ * 100)) for _ in stats[3:6]))
    print("Evaluate done")


def ap_at_ten_iou_thr_summary(coco_eval):
    print("Evaluate AP with 10 IoU thresholds")
    stats = []
    for i in range(10):
        ap = summarize(coco_eval, 1, iouThr=coco_eval.params.iouThrs[i], maxDets=coco_eval.params.maxDets[2])
        stats.append(ap)
    print("Easy copy IoU=0.5:0.1:0.9 : %s" % ", ".join(str("%2.1f" % (_ * 100)) for _ in stats[::2]))
    print("Easy copy IoU=0.5:0.05:0.95 : %s" % ", ".join(str("%2.1f" % (_ * 100)) for _ in stats))
    print("Evaluate done")

