from models.tridentnet.builder import TridentFasterRcnn as Detector
from models.tridentnet.builder_pami import TridentResNetV1bC4 as Backbone
from models.tridentnet.builder import TridentRpnHead as RpnHead
from models.tridentnet.builder import process_branch_rpn_outputs
from symbol.builder import Neck
from symbol.builder import RoiAlign as RoiExtractor
from symbol.builder import BboxC5V1Head as BboxHead
from mxnext.complicate import normalizer_factory


def get_config(is_train):
    class General:
        log_frequency = 10
        name = __name__.rsplit("/")[-1].rsplit(".")[-1]
        batch_image = 3 if is_train else 1
        fp16 = True
        multi_symbol_test = True

    class Trident:
        num_branch = 3 if is_train else 1
        train_scaleaware = False
        test_scaleaware = False
        branch_ids = range(num_branch) if is_train else [1]
        branch_dilates = [1, 2, 3] if is_train else [2]
        valid_ranges = [(0, -1), (0, -1), (0, -1)] if is_train else [(0, -1)]
        valid_ranges_on_origin = True
        branch_bn_shared = False
        branch_conv_shared = True
        branch_deform = False

        assert num_branch == len(branch_ids)
        assert num_branch == len(valid_ranges)

    class KvstoreParam:
        kvstore     = "local"
        batch_image = General.batch_image
        gpus        = [0, 1, 2, 3, 4, 5, 6, 7]
        fp16        = General.fp16

    class NormalizeParam:
        normalizer = normalizer_factory(type="syncbn", ndev=len(KvstoreParam.gpus))
        # normalizer = normalizer_factory(type="fixbn")

    class BackboneParam:
        fp16 = General.fp16
        normalizer = NormalizeParam.normalizer
        depth = 101
        num_branch = Trident.num_branch
        branch_ids = Trident.branch_ids
        branch_dilates = Trident.branch_dilates
        branch_bn_shared = Trident.branch_bn_shared
        branch_conv_shared = Trident.branch_conv_shared
        branch_deform = Trident.branch_deform
        arg_params = {}

    class NeckParam:
        fp16 = General.fp16
        normalizer = NormalizeParam.normalizer


    class RpnParam:
        fp16 = General.fp16
        normalizer = NormalizeParam.normalizer
        batch_image = General.batch_image * Trident.num_branch

        class anchor_generate:
            scale = (2, 4, 8, 16, 32)
            ratio = (0.5, 1.0, 2.0)
            stride = 16
            image_anchor = 256

        class head:
            conv_channel = 512
            mean = (0, 0, 0, 0)
            std = (1, 1, 1, 1)

        class regress_target:
            smooth_l1_scalar = 999

        class proposal:
            pre_nms_top_n = 12000 if is_train else 6000
            post_nms_top_n = 500 if is_train else 300
            nms_thr = 0.7
            min_bbox_side = 0

        class subsample_proposal:
            proposal_wo_gt = True
            image_roi = 128
            fg_fraction = 0.5
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
        fp16        = General.fp16
        normalizer  = NormalizeParam.normalizer
        num_class   = 1 + 80
        image_roi   = 128
        batch_image = General.batch_image * Trident.num_branch

        class regress_target:
            class_agnostic = True
            mean = (0.0, 0.0, 0.0, 0.0)
            std = (0.1, 0.1, 0.2, 0.2)
            smooth_l1_scalar = 999


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
        train_sym = detector.get_train_symbol(
            backbone, neck, rpn_head, roi_extractor, bbox_head,
            num_branch=Trident.num_branch, scaleaware=Trident.train_scaleaware)
        rpn_test_sym = None
        test_sym = None
    else:
        train_sym = None
        rpn_test_sym = detector.get_rpn_test_symbol(backbone, neck, rpn_head, Trident.num_branch)
        if General.multi_symbol_test:
            tag2sym = {}
            for dil in [1, 2, 3]:
                BackboneParam.branch_dilates = [dil]
                backbone = Backbone(BackboneParam)
                Detector._rpn_output = None
                rpn_head._cls_logit = None
                rpn_head._bbox_delta = None
                rpn_head._proposal = None
                bbox_head._head_feat = None
                test_sym = detector.get_test_symbol(
                    backbone, neck, rpn_head, roi_extractor, bbox_head, num_branch=Trident.num_branch)
                tag2sym[dil] = test_sym
        else:
            test_sym = detector.get_test_symbol(
                backbone, neck, rpn_head, roi_extractor, bbox_head, num_branch=Trident.num_branch)


    class ModelParam:
        train_symbol = train_sym
        test_symbol = test_sym
        tag_to_symbol = tag2sym if General.multi_symbol_test else None
        rpn_test_symbol = rpn_test_sym

        from_scratch = False
        random = True
        memonger = False
        memonger_until = "stage3_unit21_plus"
        disable_merge_bn = True

        def process_weight(sym, arg_params, aux_params):
            # for offset init
            arg_params.update(BackboneParam.arg_params)

            # for trident syncbn init
            import re
            import logging

            logger = logging.getLogger()
            # for trident non-shared initialization
            for k in sym.list_arguments():
                branch_name = re.sub('_branch\d+', '', k)
                if k != branch_name and branch_name in arg_params:
                    arg_params[k] = arg_params[branch_name]
                    logger.info('init arg {} with {}'.format(k, branch_name))

            for k in sym.list_auxiliary_states():
                branch_name = re.sub('_branch\d+', '', k)
                if k != branch_name and branch_name in aux_params:
                    aux_params[k] = aux_params[branch_name]
                    logger.info('init aux {} with {}'.format(k, branch_name))

        class pretrain:
            prefix = "pretrain_model/resnet%s_v1b" % BackboneParam.depth
            epoch = 0
            fixed_param = []


    class OptimizeParam:
        class optimizer:
            type = "sgd"
            lr = 0.01 / 8 * len(KvstoreParam.gpus) * KvstoreParam.batch_image
            momentum = 0.9
            wd = 0.0001
            clip_gradient = 5

        class schedule:
            begin_epoch = 17
            end_epoch = 18
            lr_iter = [210000 * 16 // (len(KvstoreParam.gpus) * KvstoreParam.batch_image),
                       250000 * 16 // (len(KvstoreParam.gpus) * KvstoreParam.batch_image)]

        class warmup:
            type = "gradual"
            lr = 0.0
            iter = 3000 * 16 // (len(KvstoreParam.gpus) * KvstoreParam.batch_image)


    class TestScaleParam:
        long_side = 3000
        # short_sides  = (400,      500,      600,      700,      800,     900,     1000,    1100,     1200,     1300,     1400,     1800)  # noqa: E241
        # scale_ranges = ((96, -1), (96, -1), (64, -1), (64, -1), (0, -1), (0, -1), (0, -1), (0, 256), (0, 256), (0, 192), (0, 192), (0, 96))
        # model_tags   = (2,        2,        2,        2,        2,       2,       2,       2,        2,        2,        2,        2)  # noqa: E241
        # short_sides  = (600,      800,     1000,    1200,     1400,     1800)  # noqa: E241
        # scale_ranges = ((64, -1), (0, -1), (0, -1), (0, 256), (0, 192), (0, 96))
        # model_tags   = (2,        2,       2,       2,        2,        2)  # noqa: E241
        short_sides  = (800,     1000,    )  # noqa: E241
        scale_ranges = ((0, -1), (0, -1), )
        model_tags   = (2,        2,      )  # noqa: E241
        model_tags   = model_tags if General.multi_symbol_test else None


    class TestParam:
        min_det_score = 0.001
        max_det_per_image = 100

        from models.tridentnet.builder_pami import add_scale_and_range_to_roidb, add_flip_to_roidb, compose_process_roidb
        from models.tridentnet.builder_pami import filter_bbox_by_scale_range, flip_bbox_for_output, compose_process_output
        process_roidb = compose_process_roidb(
            add_scale_and_range_to_roidb(TestScaleParam.short_sides, TestScaleParam.long_side, TestScaleParam.scale_ranges, TestScaleParam.model_tags),
            # add_flip_to_roidb
        )
        process_output = compose_process_output(
            filter_bbox_by_scale_range,
            # flip_bbox_for_output
        )

        process_rpn_output = lambda x, y: process_branch_rpn_outputs(x, Trident.num_branch)

        class model:
            prefix = "experiments/{}/checkpoint".format(General.name)
            epoch = OptimizeParam.schedule.end_epoch

        class nms:
            from operator_py.nms import soft_nms_bbox_vote_wrapper
            type = soft_nms_bbox_vote_wrapper(0.5, 0.9)

        class coco:
            annotation = "data/coco/annotations/instances_val2017.json"

    # data processing
    class NormParam:
        mean = tuple(i * 255 for i in (0.485, 0.456, 0.406))  # RGB order
        std = tuple(i * 255 for i in (0.229, 0.224, 0.225))

    class ResizeParam:
        short = 1000
        long = 1200 if is_train else 2000

    class RandResizeParam:
        short = None  # generate on the fly
        long = None
        short_ranges = [600, 800, 1000, 1200]
        long_ranges = [2000, 2000, 2000, 2000]

    class RandCropParam:
        mode = "center"  # random or center
        short = 1008
        long = 1600

    class PadParam:
        short = 1008
        long = 1600 if is_train else 2000
        max_num_gt = 100

    class ScaleRange:
        valid_ranges = Trident.valid_ranges
        cal_on_origin = Trident.valid_ranges_on_origin  # True: valid_ranges on origin image scale / valid_ranges on resized image scale

    class AnchorTarget2DParam:
        class generate:
            short = 1008 // 16
            long = 1600 // 16
            stride = 16
            scales = (2, 4, 8, 16, 32)
            aspects = (0.5, 1.0, 2.0)

        class assign:
            allowed_border = 0
            pos_thr = 0.6
            neg_thr = 0.6
            min_pos_thr = 0.0

        class sample:
            image_anchor = 256
            pos_fraction = 0.5

        class trident:
            invalid_anchor_threshd = 0.3


    class RenameParam:
        mapping = dict(image="data")


    from core.detection_input import ReadRoiRecord, Resize2DImageBbox, \
        ConvertImageFromHwcToChw, Flip2DImage, Flip2DImageBbox, Pad2DImageBbox, \
        RenameRecord, Norm2DImage, RandResize2DImageBbox, Resize2DImageByRoidb, RandCrop2DImageBbox
    from models.tridentnet.input import ScaleAwareRange, TridentAnchorTarget2D

    if is_train:
        transform = [
            ReadRoiRecord(None),
            Norm2DImage(NormParam),
            RandResize2DImageBbox(RandResizeParam),
            RandCrop2DImageBbox(RandCropParam),
            Flip2DImageBbox(),
            Pad2DImageBbox(PadParam),
            ConvertImageFromHwcToChw(),
            ScaleAwareRange(ScaleRange),
            TridentAnchorTarget2D(AnchorTarget2DParam),
            RenameRecord(RenameParam.mapping)
        ]
        data_name = ["data", "im_info", "gt_bbox"]
        if Trident.train_scaleaware:
            data_name.append("valid_ranges")
        label_name = ["rpn_cls_label", "rpn_reg_target", "rpn_reg_weight"]
    else:
        transform = [
            ReadRoiRecord(None),
            Norm2DImage(NormParam),
            # Flip2DImage(),
            Resize2DImageByRoidb(),
            ConvertImageFromHwcToChw(),
            RenameRecord(RenameParam.mapping)
        ]
        data_name = ["data", "im_info", "im_id", "rec_id"]
        label_name = ["model_tag"] if General.multi_symbol_test else []

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
