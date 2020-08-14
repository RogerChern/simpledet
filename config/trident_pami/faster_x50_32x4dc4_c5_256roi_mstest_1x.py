from symbol.builder import FasterRcnn as Detector
from models.dilated.builder import DilatedResNeXtC4 as Backbone
from symbol.builder import Neck
from symbol.builder import RpnHead
from symbol.builder import RoiAlign as RoiExtractor
from models.tridentnet.builder_pami import BboxResNeXtC5Head as BboxHead
from mxnext.complicate import normalizer_factory


def get_config(is_train):
    class General:
        log_frequency = 10
        name = __name__.rsplit("/")[-1].rsplit(".")[-1]
        batch_image = 4 if is_train else 1
        fp16 = True


    class KvstoreParam:
        kvstore     = "local"
        batch_image = General.batch_image
        gpus        = [0, 1, 2, 3, 4, 5, 6, 7]
        fp16        = General.fp16


    class NormalizeParam:
        # normalizer = normalizer_factory(type="syncbn", ndev=len(KvstoreParam.gpus))
        normalizer = normalizer_factory(type="fixbn")


    class BackboneParam:
        fp16 = General.fp16
        normalizer = NormalizeParam.normalizer
        depth = 50
        group = 32
        channel_per_group = 4
        c4_dil = 1


    class NeckParam:
        fp16 = General.fp16
        normalizer = NormalizeParam.normalizer


    class RpnParam:
        fp16 = General.fp16
        normalizer = NormalizeParam.normalizer
        batch_image = General.batch_image

        class anchor_generate:
            scale = (2, 4, 8, 16, 32)
            ratio = (0.5, 1.0, 2.0)
            stride = 16
            image_anchor = 256

        class head:
            conv_channel = 512
            mean = (0, 0, 0, 0)
            std = (1, 1, 1, 1)

        class proposal:
            pre_nms_top_n = 12000 if is_train else 6000
            post_nms_top_n = 2000 if is_train else 300
            nms_thr = 0.7
            min_bbox_side = 0

        class subsample_proposal:
            proposal_wo_gt = False
            image_roi = 256
            fg_fraction = 0.25
            fg_thr = 0.5
            bg_thr_hi = 0.5
            bg_thr_lo = 0.0

        class bbox_target:
            num_reg_class = 2
            class_agnostic = True
            weight = (1.0, 1.0, 1.0, 1.0)
            mean = (0.0, 0.0, 0.0, 0.0)
            std = (0.1, 0.1, 0.2, 0.2)


    class BboxParam:
        fp16 = General.fp16
        normalizer = NormalizeParam.normalizer
        num_class = 1 + 80
        image_roi = RpnParam.subsample_proposal.image_roi
        batch_image = General.batch_image
        group = BackboneParam.group
        channel_per_group = BackboneParam.channel_per_group

        class regress_target:
            class_agnostic = RpnParam.bbox_target.class_agnostic
            mean = RpnParam.bbox_target.mean
            std = RpnParam.bbox_target.std


    class RoiParam:
        fp16 = General.fp16
        normalizer = NormalizeParam.normalizer
        out_size = 7
        stride = 16


    class DatasetParam:
        if is_train:
            image_set = ("coco_train2017", )
        else:
            image_set = ("coco_val2017", )

    backbone = Backbone(BackboneParam)
    neck = Neck(NeckParam)
    rpn_head = RpnHead(RpnParam)
    roi_extractor = RoiExtractor(RoiParam)
    bbox_head = BboxHead(BboxParam)
    detector = Detector()
    if is_train:
        train_sym = detector.get_train_symbol(backbone, neck, rpn_head, roi_extractor, bbox_head)
        rpn_test_sym = None
        test_sym = None
    else:
        train_sym = None
        rpn_test_sym = detector.get_rpn_test_symbol(backbone, neck, rpn_head)
        test_sym = detector.get_test_symbol(backbone, neck, rpn_head, roi_extractor, bbox_head)


    class ModelParam:
        train_symbol = train_sym
        test_symbol = test_sym
        rpn_test_symbol = rpn_test_sym

        from_scratch = False
        random = True
        memonger = False
        memonger_until = "stage3_unit21_plus"

        class pretrain:
            prefix = "pretrain_model/resnext%s_32x4d" % BackboneParam.depth
            epoch = 0
            fixed_param = ["conv0", "stage1", "gamma", "beta"]


    class OptimizeParam:
        class optimizer:
            type = "sgd"
            lr = 0.01 / 8 * len(KvstoreParam.gpus) * KvstoreParam.batch_image
            momentum = 0.9
            wd = 0.0001
            clip_gradient = 5

        class schedule:
            begin_epoch = 0
            end_epoch = 6
            lr_iter = [60000 * 16 // (len(KvstoreParam.gpus) * KvstoreParam.batch_image),
                       80000 * 16 // (len(KvstoreParam.gpus) * KvstoreParam.batch_image)]

        class warmup:
            type = "gradual"
            lr = 0.0
            iter = 3000 * 16 // (len(KvstoreParam.gpus) * KvstoreParam.batch_image)


    class TestScaleParam:
        short_sides = (400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1800)
        long_side = 3000
        scale_ranges = ((96, -1), (96, -1), (64, -1), (64, -1), (0, -1), (0, -1), (0, -1), (0, 256), (0, 256), (0, 192), (0, 192), (0, 96))


    class TestParam:
        min_det_score = 0.05
        max_det_per_image = 100

        def process_roidb(roidb):
            ms_roidb = []
            for r_ in roidb:
                for short, range in zip(TestScaleParam.short_sides, TestScaleParam.scale_ranges):
                    r = r_.copy()
                    r["bbox_valid_range_on_original_input"] = range
                    r["resize_long"] = TestScaleParam.long_side
                    r["resize_short"] = short
                    ms_roidb.append(r)
            return ms_roidb

        from models.tridentnet.builder_pami import filter_bbox_by_scale_range
        process_output = filter_bbox_by_scale_range

        class model:
            prefix = "experiments/{}/checkpoint".format(General.name)
            epoch = OptimizeParam.schedule.end_epoch

        class nms:
            from operator_py.nms import soft_bbox_vote_wrapper
            type = soft_bbox_vote_wrapper(0.66, 0.05)
            # from operator_py.nms import cython_soft_nms_wrapper
            # type = cython_soft_nms_wrapper(0.5)

        class coco:
            annotation = "data/coco/annotations/instances_val2017.json"

    # data processing
    class NormParam:
        mean = tuple(i * 255 for i in (0.485, 0.456, 0.406))  # RGB order
        std = tuple(i * 255 for i in (0.229, 0.224, 0.225))


    class ResizeParam:
        short = 800
        long = 1200 if is_train else 2000


    class PadParam:
        short = 800
        long = 1200
        max_num_gt = 100


    class AnchorTarget2DParam:
        class generate:
            short = 800 // 16
            long = 1200 // 16
            stride = 16
            scales = (2, 4, 8, 16, 32)
            aspects = (0.5, 1.0, 2.0)

        class assign:
            allowed_border = 0
            pos_thr = 0.7
            neg_thr = 0.3
            min_pos_thr = 0.0

        class sample:
            image_anchor = 256
            pos_fraction = 0.5


    class RenameParam:
        mapping = dict(image="data")


    from core.detection_input import ReadRoiRecord, Resize2DImageBbox, \
        ConvertImageFromHwcToChw, Flip2DImageBbox, Pad2DImageBbox, \
        RenameRecord, AnchorTarget2D, Norm2DImage, Resize2DImageByRoidb

    if is_train:
        transform = [
            ReadRoiRecord(None),
            Norm2DImage(NormParam),
            Resize2DImageBbox(ResizeParam),
            Flip2DImageBbox(),
            Pad2DImageBbox(PadParam),
            ConvertImageFromHwcToChw(),
            AnchorTarget2D(AnchorTarget2DParam),
            RenameRecord(RenameParam.mapping)
        ]
        data_name = ["data", "im_info", "gt_bbox"]
        label_name = ["rpn_cls_label", "rpn_reg_target", "rpn_reg_weight"]
    else:
        transform = [
            ReadRoiRecord(None),
            Norm2DImage(NormParam),
            Resize2DImageByRoidb(),
            # Resize2DImageBbox(),
            ConvertImageFromHwcToChw(),
            RenameRecord(RenameParam.mapping)
        ]
        data_name = ["data", "im_info", "im_id", "rec_id"]
        label_name = []

    import core.detection_metric as metric

    rpn_acc_metric = metric.AccWithIgnore(
        "RpnAcc",
        ["rpn_cls_loss_output"],
        ["rpn_cls_label"]
    )
    rpn_l1_metric = metric.L1(
        "RpnL1",
        ["rpn_reg_loss_output"],
        ["rpn_cls_label"]
    )
    # for bbox, the label is generated in network so it is an output
    box_acc_metric = metric.AccWithIgnore(
        "RcnnAcc",
        ["bbox_cls_loss_output", "bbox_label_blockgrad_output"],
        []
    )
    box_l1_metric = metric.L1(
        "RcnnL1",
        ["bbox_reg_loss_output", "bbox_label_blockgrad_output"],
        []
    )

    metric_list = [rpn_acc_metric, rpn_l1_metric, box_acc_metric, box_l1_metric]

    return General, KvstoreParam, RpnParam, RoiParam, BboxParam, DatasetParam, \
        ModelParam, OptimizeParam, TestParam, \
        transform, data_name, label_name, metric_list
