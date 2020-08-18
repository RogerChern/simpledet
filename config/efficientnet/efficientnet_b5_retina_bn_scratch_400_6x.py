from models.retinanet.builder import RetinaNet as Detector
from models.efficientnet.builder import EfficientNetB5FPN as Backbone
from models.retinanet.builder import RetinaNetNeck as Neck
from models.retinanet.builder import RetinaNetHead as RpnHead
from mxnext.complicate import normalizer_factory


def get_config(is_train):
    class General:
        log_frequency = 10
        name = __name__.rsplit("/")[-1].rsplit(".")[-1]
        batch_image = 8 if is_train else 1
        fp16 = True
        loader_worker = 8


    class KvstoreParam:
        kvstore     = "nccl"
        batch_image = General.batch_image
        gpus        = [0, 1, 2, 3, 4, 5, 6, 7]
        fp16        = General.fp16


    class NormalizeParam:
        normalizer = normalizer_factory(type="localbn", ndev=len(KvstoreParam.gpus))
        # normalizer = normalizer_factory(type="gn")


    class BackboneParam:
        fp16 = General.fp16
        normalizer = NormalizeParam.normalizer


    class NeckParam:
        fp16 = General.fp16
        normalizer = NormalizeParam.normalizer


    class RpnParam:
        fp16 = General.fp16
        normalizer = NormalizeParam.normalizer
        batch_image = General.batch_image
        num_class   = 1 + 80


        class anchor_generate:
            scale = (4 * 2 ** 0, 4 * 2 ** (1.0 / 3.0), 4 * 2 ** (2.0 / 3.0))
            ratio = (0.5, 1.0, 2.0)
            stride = (8, 16, 32, 64, 128)
            image_anchor = None

        class head:
            conv_channel = 256
            mean = (0, 0, 0, 0)
            std = (1, 1, 1, 1)

        class proposal:
            pre_nms_top_n = 1000
            post_nms_top_n = None
            nms_thr = None
            min_bbox_side = None
            min_det_score = 0.05 # filter score in network

        class focal_loss:
            alpha = 0.25
            gamma = 2.0


    class BboxParam:
        pass


    class RoiParam:
        pass


    class DatasetParam:
        if is_train:
            image_set = ("coco_train2017", )
            total_image = 82783 + 35504
        else:
            image_set = ("coco_val2017", )
            total_image = 5000

    backbone = Backbone(BackboneParam)
    neck = Neck(NeckParam)
    rpn_head = RpnHead(RpnParam)
    detector = Detector()
    if is_train:
        train_sym = detector.get_train_symbol(backbone, neck, rpn_head)
        test_sym = None
    else:
        train_sym = None
        test_sym = detector.get_test_symbol(backbone, neck, rpn_head)



    class ModelParam:
        train_symbol = train_sym
        test_symbol = test_sym

        from_scratch = True
        random = True
        memonger = False
        memonger_until = "stage3_unit21_plus"

        class pretrain:
            prefix = None
            epoch = 0
            fixed_param = []


    class OptimizeParam:
        class optimizer:
            type = "sgd"
            lr = 0.005 / 8 * len(KvstoreParam.gpus) * KvstoreParam.batch_image
            momentum = 0.9
            wd = 1e-4
            clip_gradient = None

        class schedule:
            mult = 6
            begin_epoch = 0
            end_epoch = 6 * mult
            if mult <= 2:
                lr_iter = [60000 * mult * 16 // (len(KvstoreParam.gpus) * KvstoreParam.batch_image),
                           80000 * mult * 16 // (len(KvstoreParam.gpus) * KvstoreParam.batch_image)]
            else:
                # follow the setting in Rethinking ImageNet Pre-training
                # reduce the lr in the last 60k and 20k iterations
                lr_iter = [(DatasetParam.total_image * 2 // 16 * end_epoch - 60000) * 16 //
                    (len(KvstoreParam.gpus) * KvstoreParam.batch_image),
                    (DatasetParam.total_image * 2 // 16 * end_epoch - 20000) * 16 //
                    (len(KvstoreParam.gpus) * KvstoreParam.batch_image)]

        class warmup:
            type = "gradual"
            lr = 0
            iter = 500


    class TestParam:
        min_det_score = 0.05
        max_det_per_image = 100

        process_roidb = lambda x: x
        process_output = lambda x, y: x

        class model:
            prefix = "experiments/{}/checkpoint".format(General.name)
            epoch = OptimizeParam.schedule.end_epoch

        class nms:
            type = "nms"
            thr = 0.5

        class coco:
            annotation = "data/coco/annotations/instances_val2017.json"


    # data processing
    class NormParam:
        mean = tuple(i * 255 for i in (0.485, 0.456, 0.406)) # RGB order
        std = tuple(i * 255 for i in (0.229, 0.224, 0.225))

    # data processing
    class ResizeParam:
        short = 400
        long = 600


    class PadParam:
        short = 400
        long = 600
        max_num_gt = 100


    class AnchorTarget2DParam:
        def __init__(self):
            self.generate = self._generate()

        class _generate:
            def __init__(self):
                self.stride = (8, 16, 32, 64, 128)
                self.short = (50, 25, 13, 7, 4)
                self.long = (75, 38, 19, 10, 5)
            scales = (4 * 2 ** 0, 4 * 2 ** (1.0 / 3.0), 4 * 2 ** (2.0 / 3.0))
            aspects = (0.5, 1.0, 2.0)

        class assign:
            allowed_border = 9999
            pos_thr = 0.5
            neg_thr = 0.4
            min_pos_thr = 0.0

        class sample:
            image_anchor = None
            pos_fraction = None


    class RenameParam:
        mapping = dict(image="data")


    from core.detection_input import ReadRoiRecord, Resize2DImageBbox, \
        ConvertImageFromHwcToChw, Flip2DImageBbox, Pad2DImageBbox, \
        RenameRecord
    from models.retinanet.input import PyramidAnchorTarget2D, Norm2DImage

    if is_train:
        transform = [
            ReadRoiRecord(None),
            Norm2DImage(NormParam),
            Resize2DImageBbox(ResizeParam),
            Flip2DImageBbox(),
            Pad2DImageBbox(PadParam),
            ConvertImageFromHwcToChw(),
            PyramidAnchorTarget2D(AnchorTarget2DParam()),
            RenameRecord(RenameParam.mapping)
        ]
        data_name = ["data"]
        label_name = ["rpn_cls_label", "rpn_reg_target", "rpn_reg_weight"]
    else:
        transform = [
            ReadRoiRecord(None),
            Norm2DImage(NormParam),
            Resize2DImageBbox(ResizeParam),
            ConvertImageFromHwcToChw(),
            RenameRecord(RenameParam.mapping)
        ]
        data_name = ["data", "im_info", "im_id", "rec_id"]
        label_name = []

    from models.retinanet import metric

    rpn_acc_metric = metric.FGAccMetric(
        "FGAcc",
        ["cls_loss_output"],
        ["rpn_cls_label"]
    )

    metric_list = [rpn_acc_metric]

    return General, KvstoreParam, RpnParam, RoiParam, BboxParam, DatasetParam, \
           ModelParam, OptimizeParam, TestParam, \
           transform, data_name, label_name, metric_list
