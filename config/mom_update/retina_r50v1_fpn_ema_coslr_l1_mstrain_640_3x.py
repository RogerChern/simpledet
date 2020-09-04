from models.retinanet.builder import RetinaNet as Detector
from models.retinanet.builder import MSRAResNet50V1FPN as Backbone
from models.retinanet.builder import RetinaNetNeck as Neck
from models.retinanet.builder import RetinaNetHead as RpnHead
from mxnext.complicate import normalizer_factory


def get_config(is_train):
    class General:
        log_frequency = 10
        name = __name__.rsplit("/")[-1].rsplit(".")[-1]
        batch_image = 4 if is_train else 1
        fp16 = True


    class KvstoreParam:
        kvstore     = "nccl"
        batch_image = General.batch_image
        gpus        = [0, 1, 2, 3]
        fp16        = General.fp16


    class NormalizeParam:
        normalizer = normalizer_factory(type="fixbn")


    class BackboneParam:
        fp16 = General.fp16
        normalizer = NormalizeParam.normalizer


    class NeckParam:
        fp16 = General.fp16
        normalizer = NormalizeParam.normalizer


    class RpnParam:
        num_class = 1 + 80
        fp16 = General.fp16
        normalizer = NormalizeParam.normalizer
        batch_image = General.batch_image

        class anchor_generate:
            scale = (4 * 2 ** 0, 4 * 2 ** (1.0 / 3.0), 4 * 2 ** (2.0 / 3.0))
            ratio = (0.5, 1.0, 2.0)
            stride = (8, 16, 32, 64, 128)

        class head:
            conv_channel = 256

        class proposal:
            pre_nms_top_n = 1000
            min_det_score = 0.05  # filter score in network

        class focal_loss:
            alpha = 0.25
            gamma = 2.0

        class regress_target:
            smooth_l1_scalar = 999


    class BboxParam:
        pass

    class RoiParam:
        pass

    class DatasetParam:
        if is_train:
            image_set = ("coco_train2017", )
        else:
            image_set = ("coco_val2017", )

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

        from_scratch = False
        random = True
        memonger = False
        memonger_until = "stage3_unit21_plus"

        class pretrain:
            prefix = "pretrain_model/resnet-v1-50"
            epoch = 0
            fixed_param = ["conv0", "stage1", "gamma", "beta"]


    class OptimizeParam:
        class optimizer:
            type = "sgd"
            lr = 0.005 / 8 * len(KvstoreParam.gpus) * KvstoreParam.batch_image
            momentum = 0.9
            wd = 0.0001
            clip_gradient = None
            lr_mode = "cosine"

        class schedule:
            begin_epoch = 0
            end_epoch = 18
            lr_iter = []

        class warmup:
            type = "gradual"
            lr = 0.005 / 8 * len(KvstoreParam.gpus) * KvstoreParam.batch_image / 3
            iter = 1000 * 16 // (len(KvstoreParam.gpus) * KvstoreParam.batch_image)

        class ema:
            momentum = [0.999, 0.9995, 0.9999, 0.99995, 0.99999, 0.999995]
            zero_init = False
            save_iter = 5000 * 16 // (len(KvstoreParam.gpus) * KvstoreParam.batch_image)

    class TestParam:
        min_det_score = 0  # filter appended boxes
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
        mean = (122.7717, 115.9465, 102.9801)  # RGB order
        std = (1.0, 1.0, 1.0)


    class ResizeParam:
        short = 800
        long = 1333


    class PadParam:
        short = 800
        long = 1333
        max_num_gt = 100


    class RandResizeParam:
        short = None  # generate on the fly
        long = None
        short_ranges = [640, 672, 704, 736, 768, 800]
        long_ranges = [2000, 2000, 2000, 2000, 2000, 2000]


    class RandCropParam:
        mode = "center"  # random or center
        short = 800
        long = 1333


    class AnchorTarget2DParam:
        def __init__(self):
            self.generate = self._generate()

        class _generate:
            def __init__(self):
                self.short = (100, 50, 25, 13, 7)
                self.long = (167, 84, 42, 21, 11)
                self.stride = (8, 16, 32, 64, 128)

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
        RenameRecord, RandResize2DImageBbox, RandCrop2DImageBbox
    from models.retinanet.input import PyramidAnchorTarget2D, Norm2DImage

    if is_train:
        transform = [
            ReadRoiRecord(None),
            Norm2DImage(NormParam),
            RandResize2DImageBbox(RandResizeParam),
            RandCrop2DImageBbox(RandCropParam),
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
