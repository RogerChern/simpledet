import numpy as np

from core.detection_input import DetectionAugmentation


class Pad2DImageBboxRbbox(DetectionAugmentation):
    """
    input: image, ndarray(h, w, rgb)
           gt_bbox, ndarry(n, 5)
           gt_rbbox, ndarray(n, 5)
    output: image, ndarray(h, w, rgb)
            gt_bbox, ndarray(max_num_gt, 5)
            gt_rbbox, ndarray(max_num_gt, 5)
    """

    def __init__(self, pPad):
        super().__init__()
        self.p = pPad  # type: PadParam

    def apply(self, input_record):
        p = self.p

        image = input_record["image"]
        gt_bbox = input_record["gt_bbox"]
        gt_rbbox = input_record["gt_rbbox"]

        h, w = image.shape[:2]
        shape = (p.long, p.short, 3) if input_record["orientation"] == "vertical" \
            else (p.short, p.long, 3)

        padded_image = np.zeros(shape, dtype=np.float32)
        padded_image[:h, :w] = image
        padded_gt_bbox = np.full(shape=(p.max_num_gt, 5), fill_value=-1, dtype=np.float32)
        padded_gt_bbox[:len(gt_bbox)] = gt_bbox
        padded_gt_rbbox = np.full(shape=(p.max_num_gt, 5), fill_value=-1, dtype=np.float32)
        padded_gt_rbbox[:len(gt_rbbox)] = gt_rbbox

        input_record["image"] = padded_image
        input_record["gt_bbox"] = padded_gt_bbox
        input_record["gt_rbbox"] = padded_gt_rbbox
