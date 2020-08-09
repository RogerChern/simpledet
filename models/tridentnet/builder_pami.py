from collections.abc import Iterable
import mxnet as mx
import mxnext as X
from mxnext import conv, relu, add
from mxnext.backbone import resnet_v1b_helper, resnet_v1d_helper
from symbol.builder import Backbone
from models.tridentnet.builder import TridentFasterRcnn
from models.tridentnet.resnet_v2 import TridentResNetV2Builder


def affine_conv(data, name, filter, kernel=1, stride=1, pad=None, dilate=1, rotate=None,
         num_group=1, no_bias=True, init=None, lr_mult=1.0, wd_mult=1.0, weight=None, bias=None,
         arg_params=None):
    assert stride == 1

    import numpy as np

    def get_2d_rotate_matrix(radian):
        return np.array([
            [np.cos(radian), -np.sin(radian)],
            [np.sin(radian),  np.cos(radian)]])

    def generate_affine_offset(dilate, rotate):
        original_coord = [-1, -1, -1, 0, -1, 1, 0, -1, 0, 0, 0, 1, 1, -1, 1, 0, 1, 1]
        original_coord = np.array(original_coord)
        original_coord = original_coord.reshape(9, 2)
        transformed_coord = get_2d_rotate_matrix(rotate) @ (original_coord.T)
        transformed_coord = transformed_coord.T
        transformed_coord[:, 0] *= dilate[1]
        transformed_coord[:, 1] *= dilate[0]
        offset = transformed_coord - original_coord
        offset = mx.nd.array(offset, ctx=mx.cpu()).reshape(1, 18, 1, 1)
        arg_params[name + "_single_offset"] = offset

    if not isinstance(kernel, Iterable):
        kernel = (kernel, kernel)
    assert kernel == (3, 3), "only support (3, 3) kernel currently"
    if not isinstance(stride, Iterable):
        stride = (stride, stride)
    if not isinstance(dilate, Iterable):
        dilate = (dilate, dilate)
    if pad is None:
        assert kernel[0] % 2 == 1, "Specify pad for an even kernel size for {}".format(name)
        assert kernel[1] % 2 == 1, "Specify pad for an even kernel size for {}".format(name)
        pad = (((kernel[0] - 1) * dilate[0] + 1) // 2, ((kernel[1] - 1) * dilate[1] + 1) // 2)
    if not isinstance(pad, Iterable):
        pad = (pad, pad)

    # specific initialization method
    if not isinstance(weight, mx.sym.Symbol):
        if init is not None:
            assert isinstance(init, mx.init.Initializer)
            weight = mx.sym.var(name=name + "_weight", init=init, lr_mult=lr_mult, wd_mult=wd_mult)
        elif lr_mult != 1.0 or wd_mult != 1.0:
            weight = mx.sym.var(name=name + "_weight", lr_mult=lr_mult, wd_mult=wd_mult)
        else:
            weight = None

    # get the affine conv offset
    generate_affine_offset(dilate, rotate)
    single_offset = X.var(shape=(1, 18, 1, 1), name=name + "_single_offset")
    offset = mx.sym.ones(shape=(2, 18, 300, 300), dtype=np.float32, name=name + "_base_offset_grid")
    offset = mx.sym.Crop(offset, data)
    offset = mx.sym.broadcast_like(single_offset, offset)
    offset = mx.sym.stop_gradient(offset, name=name + "_offset")

    return mx.sym.contrib.DeformableConvolution(
        data=data,
        offset=offset,
        name=name,
        weight=weight,
        bias=bias,
        num_filter=filter,
        kernel=kernel,
        stride=stride,
        pad=(1, 1),
        dilate=(1, 1),
        num_group=num_group,
        workspace=512,
        no_bias=no_bias
    )


def trident_resnet_v1b_unit(input, name, id, filter, stride, dilate, proj, **kwargs):
    """
    Compared with v1, v1b moves stride=2 to the 3x3 conv instead of the 1x1 conv and use std in pre-processing
    This is also known as the facebook re-implementation of ResNet(a.k.a. the torch ResNet)
    """
    p = kwargs["params"]
    share_bn = p.branch_bn_shared
    share_conv = p.branch_conv_shared
    norm = p.normalizer

    use_affine_conv = "rotate" in kwargs and kwargs["rotate"] is not None
    if use_affine_conv:
        rotate = kwargs["rotate"]
        arg_params = p.arg_params
        assert p is not None

    ######################### prepare names #########################
    if id is not None:
        conv_postfix = ("_shared%s" if share_conv else "_branch%s") % id
        bn_postfix = ("_shared%s" if share_bn else "_branch%s") % id
        other_postfix = "_branch%s" % id
    else:
        conv_postfix = ""
        bn_postfix = ""
        other_postfix = ""

    ######################### prepare parameters #########################
    def conv_params(x):
        return dict(
            weight=X.shared_var(name + "_%s_weight" % x) if share_conv else None,
            name=name + "_%s" % x + conv_postfix
        )

    def bn_params(x):
        ret = dict(
            gamma=X.shared_var(name + "_%s_gamma" % x) if share_bn else None,
            beta=X.shared_var(name + "_%s_beta" % x) if share_bn else None,
            moving_mean=X.shared_var(name + "_%s_moving_mean" % x) if share_bn else None,
            moving_var=X.shared_var(name + "_%s_moving_var" % x) if share_bn else None,
            name=name + "_%s" % x + bn_postfix
        )
        if norm.__name__ == "gn":
            del ret["moving_mean"], ret["moving_var"]
        return ret

    ######################### construct graph #########################
    conv1 = conv(input, filter=filter // 4, **conv_params("conv1"))
    bn1 = norm(conv1, **bn_params("bn1"))
    relu1 = relu(bn1, name=name + other_postfix)

    if use_affine_conv:
        conv2 = affine_conv(relu1, filter=filter // 4, kernel=3, stride=stride, dilate=dilate, rotate=rotate,
            arg_params=arg_params, **conv_params("conv2"))
    else:
        conv2 = conv(relu1, filter=filter // 4, kernel=3, stride=stride, dilate=dilate, **conv_params("conv2"))
    bn2 = norm(conv2, **bn_params("bn2"))
    relu2 = relu(bn2, name=name + other_postfix)

    conv3 = conv(relu2, filter=filter, **conv_params("conv3"))
    bn3 = norm(conv3, **bn_params("bn3"))

    if proj:
        shortcut = conv(input, filter=filter, stride=stride, **conv_params("sc"))
        shortcut = norm(shortcut, **bn_params("sc_bn"))
    else:
        shortcut = input

    eltwise = add(bn3, shortcut, name=name + "_plus" + other_postfix)

    return relu(eltwise, name=name + "_relu" + other_postfix)


def trident_resnet_v1b_deform_unit(input, name, id, filter, stride, dilate, proj, **kwargs):
    """
    Compared with v1, v1b moves stride=2 to the 3x3 conv instead of the 1x1 conv and use std in pre-processing
    This is also known as the facebook re-implementation of ResNet(a.k.a. the torch ResNet)
    """
    p = kwargs["params"]
    share_bn = p.branch_bn_shared
    share_conv = p.branch_conv_shared
    norm = p.normalizer

    use_affine_conv = "rotate" in kwargs and kwargs["rotate"] is not None
    assert not use_affine_conv, "affine conv is not supported for deformable conv"

    ######################### prepare names #########################
    if id is not None:
        conv_postfix = ("_shared%s" if share_conv else "_branch%s") % id
        bn_postfix = ("_shared%s" if share_bn else "_branch%s") % id
        other_postfix = "_branch%s" % id
    else:
        conv_postfix = ""
        bn_postfix = ""
        other_postfix = ""

    ######################### prepare parameters #########################
    conv_params = lambda x: dict(
        weight=X.shared_var(name + "_%s_weight" % x) if share_conv else None,
        name=name + "_%s" % x + conv_postfix
    )

    def bn_params(x):
        ret = dict(
            gamma=X.shared_var(name + "_%s_gamma" % x) if share_bn else None,
            beta=X.shared_var(name + "_%s_beta" % x) if share_bn else None,
            moving_mean=X.shared_var(name + "_%s_moving_mean" % x) if share_bn else None,
            moving_var=X.shared_var(name + "_%s_moving_var" % x) if share_bn else None,
            name=name + "_%s" % x + bn_postfix
        )
        if norm.__name__ == "gn":
            del ret["moving_mean"], ret["moving_var"]
        return ret

    ######################### construct graph #########################
    conv1 = conv(input, filter=filter // 4, **conv_params("conv1"))
    bn1 = norm(conv1, **bn_params("bn1"))
    relu1 = relu(bn1, name=name + other_postfix)

    conv2_offset = conv(relu1, name=name + "_conv2_offset" + other_postfix, filter=72, kernel=3, stride=stride, dilate=dilate)
    conv2 = mx.sym.contrib.DeformableConvolution(relu1, conv2_offset, kernel=(3, 3),
        stride=(stride, stride), dilate=(dilate, dilate), pad=(dilate, dilate), num_filter=filter // 4,
        num_deformable_group=4, no_bias=True, **conv_params("conv2"))
    bn2 = norm(conv2, **bn_params("bn2"))
    relu2 = relu(bn2, name=name + other_postfix)

    conv3 = conv(relu2, filter=filter, **conv_params("conv3"))
    bn3 = norm(conv3, **bn_params("bn3"))

    if proj:
        shortcut = conv(input, filter=filter, stride=stride, **conv_params("sc"))
        shortcut = norm(shortcut, **bn_params("sc_bn"))
    else:
        shortcut = input

    eltwise = add(bn3, shortcut, name=name + "_plus" + other_postfix)

    return relu(eltwise, name=name + "_relu" + other_postfix)


def trident_resnet_v1d_unit(input, name, id, filter, stride, dilate, proj, **kwargs):
    """
    Compared with v1b, v1d
    1. replace 7x7 conv stem with 3x 3x3 convs(not reflected in this funtion)
    2. avgpool before downsampling
    """
    p = kwargs["params"]
    share_bn = p.branch_bn_shared
    share_conv = p.branch_conv_shared
    norm = p.normalizer

    use_affine_conv = "rotate" in kwargs and kwargs["rotate"] is not None
    if use_affine_conv:
        rotate = kwargs["rotate"]
        arg_params = p.arg_params
        assert p is not None

    ######################### prepare names #########################
    # if no branch id is provided, act like a normal resnet unit
    if id is not None:
        conv_postfix = ("_shared%s" if share_conv else "_branch%s") % id
        bn_postfix = ("_shared%s" if share_bn else "_branch%s") % id
        other_postfix = "_branch%s" % id
    else:
        conv_postfix = ""
        bn_postfix = ""
        other_postfix = ""

    ######################### prepare parameters #########################
    def conv_params(x):
        return dict(
            weight=X.shared_var(name + "_%s_weight" % x) if share_conv else None,
            name=name + "_%s" % x + conv_postfix
        )

    def bn_params(x):
        ret = dict(
            gamma=X.shared_var(name + "_%s_gamma" % x) if share_bn else None,
            beta=X.shared_var(name + "_%s_beta" % x) if share_bn else None,
            moving_mean=X.shared_var(name + "_%s_moving_mean" % x) if share_bn else None,
            moving_var=X.shared_var(name + "_%s_moving_var" % x) if share_bn else None,
            name=name + "_%s" % x + bn_postfix
        )
        if norm.__name__ == "gn":
            del ret["moving_mean"], ret["moving_var"]
        return ret

    ######################### construct graph #########################
    conv1 = conv(input, filter=filter // 4, **conv_params("conv1"))
    bn1 = norm(conv1, **bn_params("bn1"))
    relu1 = relu(bn1, name=name + other_postfix)

    if use_affine_conv:
        conv2 = affine_conv(relu1, filter=filter // 4, kernel=3, stride=stride, dilate=dilate, rotate=rotate,
            arg_params=arg_params, **conv_params("conv2"))
    else:
        conv2 = conv(relu1, filter=filter // 4, kernel=3, stride=stride, dilate=dilate, **conv_params("conv2"))
    bn2 = norm(conv2, **bn_params("bn2"))
    relu2 = relu(bn2, name=name + other_postfix)

    conv3 = conv(relu2, filter=filter, **conv_params("conv3"))
    bn3 = norm(conv3, **bn_params("bn3"))

    if proj:
        if not name.startswith("stage1"):
            shortcut = X.avg_pool(input, stride=stride, pad=0, pooling_convention="full", name=name + "_avgpool")
        else:
            shortcut = input
        shortcut = conv(shortcut, name=name + "_sc", filter=filter)
        shortcut = norm(shortcut, name=name + "_sc_bn")
    else:
        shortcut = input

    eltwise = add(bn3, shortcut, name=name + "_plus" + other_postfix)

    return relu(eltwise, name=name + "_relu" + other_postfix)


def trident_resnet_v1d_deform_unit(input, name, id, filter, stride, dilate, proj, **kwargs):
    """
    Compared with v1b, v1d
    1. replace 7x7 conv stem with 3x 3x3 convs(not reflected in this funtion)
    2. avgpool before downsampling
    """
    p = kwargs["params"]
    share_bn = p.branch_bn_shared
    share_conv = p.branch_conv_shared
    norm = p.normalizer

    use_affine_conv = "rotate" in kwargs and kwargs["rotate"] is not None
    assert not use_affine_conv, "affine conv is not supported for deformable conv"

    ######################### prepare names #########################
    # if no branch id is provided, act like a normal resnet unit
    if id is not None:
        conv_postfix = ("_shared%s" if share_conv else "_branch%s") % id
        bn_postfix = ("_shared%s" if share_bn else "_branch%s") % id
        other_postfix = "_branch%s" % id
    else:
        conv_postfix = ""
        bn_postfix = ""
        other_postfix = ""

    ######################### prepare parameters #########################
    conv_params = lambda x: dict(
        weight=X.shared_var(name + "_%s_weight" % x) if share_conv else None,
        name=name + "_%s" % x + conv_postfix
    )

    def bn_params(x):
        ret = dict(
            gamma=X.shared_var(name + "_%s_gamma" % x) if share_bn else None,
            beta=X.shared_var(name + "_%s_beta" % x) if share_bn else None,
            moving_mean=X.shared_var(name + "_%s_moving_mean" % x) if share_bn else None,
            moving_var=X.shared_var(name + "_%s_moving_var" % x) if share_bn else None,
            name=name + "_%s" % x + bn_postfix
        )
        if norm.__name__ == "gn":
            del ret["moving_mean"], ret["moving_var"]
        return ret

    ######################### construct graph #########################
    conv1 = conv(input, filter=filter // 4, **conv_params("conv1"))
    bn1 = norm(conv1, **bn_params("bn1"))
    relu1 = relu(bn1, name=name + other_postfix)

    conv2_offset = conv(relu1, name=name + "_conv2_offset" + other_postfix, filter=72, kernel=3, stride=stride, dilate=dilate)
    conv2 = mx.sym.contrib.DeformableConvolution(relu1, conv2_offset, kernel=(3, 3),
        stride=(stride, stride), dilate=(dilate, dilate), pad=(dilate, dilate), num_filter=filter // 4,
        num_deformable_group=4, no_bias=True, **conv_params("conv2"))
    bn2 = norm(conv2, **bn_params("bn2"))
    relu2 = relu(bn2, name=name + other_postfix)

    conv3 = conv(relu2, filter=filter, **conv_params("conv3"))
    bn3 = norm(conv3, **bn_params("bn3"))

    if proj:
        if not name.startswith("stage1"):
            shortcut = X.avg_pool(input, stride=stride, pad=0, pooling_convention="full", name=name + "_avgpool")
        else:
            shortcut = input
        shortcut = conv(shortcut, name=name + "_sc", filter=filter)
        shortcut = norm(shortcut, name=name + "_sc_bn")
    else:
        shortcut = input

    eltwise = add(bn3, shortcut, name=name + "_plus" + other_postfix)

    return relu(eltwise, name=name + "_relu" + other_postfix)


def get_trident_resnet_backbone(trident_unit, trident_deform_unit, helper):
    def build_trident_stage(input, num_block, num_tri, filter, stride, prefix, p):
        # construct leading res blocks
        data = input
        for i in range(1, num_block - num_tri + 1):
            data = trident_unit(
                input=data,
                name="%s_unit%s" % (prefix, i),
                id=None,
                filter=filter,
                stride=stride if i == 1 else 1,
                proj=True if i == 1 else False,
                dilate=1,
                params=p)

        # construct parallel branches
        cs = []
        if p.branch_rotates is None:
            rotates = [None] * len(p.branch_dilates)
        else:
            rotates = p.branch_rotates
        for dil, rot, id in zip(p.branch_dilates, rotates, p.branch_ids):
            c = data  # reset c to the output of last stage
            for i in range(num_block - num_tri + 1, num_block + 1):
                if p.branch_deform and i >= num_block - 2:
                    # convert last 3 blocks into deformable conv
                    c = trident_deform_unit(
                        input=c,
                        name="%s_unit%s" % (prefix, i),
                        id=id,
                        filter=filter,
                        stride=stride if i == 1 else 1,
                        proj=True if i == 1 else False,
                        dilate=dil,
                        rotate=rot,
                        params=p)
                else:
                    c = trident_unit(
                        input=c,
                        name="%s_unit%s" % (prefix, i),
                        id=id,
                        filter=filter,
                        stride=stride if i == 1 else 1,
                        proj=True if i == 1 else False,
                        dilate=dil,
                        rotate=rot,
                        params=p)
            cs.append(c)
        # stack branch outputs on the batch dimension
        c = mx.sym.stack(*cs, axis=1)
        c = mx.sym.reshape(c, shape=(-3, -2))
        return c


    class TridentResNetC4(Backbone):
        def __init__(self, pBackbone):
            super().__init__(pBackbone)
            p = self.p

            num_c2, num_c3, num_c4, _ = helper.depth_config[p.depth]
            branch_stage = p.branch_stage or 4
            num_tri = eval("p.num_c%d_block" % branch_stage) or (eval("num_c%d" % branch_stage) - 1)

            ################### construct symbolic graph ###################
            data = X.var("data")
            if p.fp16:
                data = data.astype("float16")
            c1 = helper.resnet_c1(data, p.normalizer)

            if branch_stage == 2:
                if p.fp16 and p.branch_rotates is not None:
                    c1 = X.to_fp32(c1, "stem_last_tofp32")
                    c2 = build_trident_stage(c1, num_c2, num_tri, 256, 1, "stage1", p)
                    c2 = X.to_fp16(c2, "stage1_last_tofp16")
                else:
                    c2 = build_trident_stage(c1, num_c2, num_tri, 256, 1, "stage1", p)
                c3 = helper.resnet_c3(c2, num_c3, 2, 1, p.normalizer)
                c4 = helper.resnet_c4(c3, num_c4, 2, 1, p.normalizer)

            elif branch_stage == 3:
                c2 = helper.resnet_c2(c1, num_c2, 1, 1, p.normalizer)
                if p.fp16 and p.branch_rotates is not None:
                    c2 = X.to_fp32(c2, "stage1_last_tofp32")
                    c3 = build_trident_stage(c2, num_c3, num_tri, 512, 2, "stage2", p)
                    c3 = X.to_fp16(c3, "stage2_last_tofp16")
                else:
                    c3 = build_trident_stage(c2, num_c3, num_tri, 512, 2, "stage2", p)
                c4 = helper.resnet_c4(c3, num_c4, 2, 1, p.normalizer)

            elif branch_stage == 4:
                c2 = helper.resnet_c2(c1, num_c2, 1, 1, p.normalizer)
                c3 = helper.resnet_c3(c2, num_c3, 2, 1, p.normalizer)
                if p.fp16 and p.branch_rotates is not None:
                    c3 = X.to_fp32(c3, "stage2_last_tofp32")
                    c4 = build_trident_stage(c3, num_c4, num_tri, 1024, 2, "stage3", p)
                    c4 = X.to_fp16(c4, "stage3_last_tofp16")
                else:
                    c4 = build_trident_stage(c3, num_c4, num_tri, 1024, 2, "stage3", p)

            else:
                raise ValueError("Unknown branch stage: %d" % branch_stage)

            self.symbol = c4

        def get_rpn_feature(self):
            return self.symbol

        def get_rcnn_feature(self):
            return self.symbol

    return TridentResNetC4


TridentResNetV1bC4 = get_trident_resnet_backbone(trident_resnet_v1b_unit, trident_resnet_v1b_deform_unit, resnet_v1b_helper)
TridentResNetV1dC4 = get_trident_resnet_backbone(trident_resnet_v1d_unit, trident_resnet_v1d_deform_unit, resnet_v1d_helper)


class OFAHead:
    def __init__(self, pOFA):
        super().__init__()
        self.p = pOFA
        self._student_feat = None

    def _transform_student_feat(self, student_feat, direct_match, use_relu_in_transform, target_channel, prefix):
        if not direct_match:
            student_hint = X.conv(student_feat, "%s_student_hint_conv" % prefix, target_channel)
            if use_relu_in_transform:
                student_hint = X.relu(student_hint, "%s_student_hint_relu" % prefix)
        return student_hint

    def get_loss(self, rcnn_feat):
        """
        Args:
            rcnn_feat: symbol for rcnn feature maps
        Returns:
            ofa_loss: symbol for once-for-all loss
        """
        to_fp32 = X.to_fp32 if self.p.fp16 else lambda x: x
        direct_match = self.p.direct_match or False
        use_relu_in_transform = self.p.use_relu_in_transform or False
        grad_scale_list = self.p.grad_scales
        target_channel_list = self.p.target_channels

        student_feat_list = [to_fp32(rcnn_feat.get_internals()[endpoint], 'ofa_s_%s_fp32' % i) for i, endpoint in enumerate(self.p.student_endpoints, start=1)]
        teacher_feat_list = [to_fp32(rcnn_feat.get_internals()[endpoint], 'ofa_t_%s_fp32' % i) for i, endpoint in enumerate(self.p.teacher_endpoints, start=1)]

        ofa_loss_list = []
        for i in range(len(grad_scale_list)):
            grad_scale = grad_scale_list[i]
            student_feat = student_feat_list[i]
            teacher_feat = teacher_feat_list[i]
            target_channel = target_channel_list[i]

            student_feat = self._transform_student_feat(student_feat, direct_match, use_relu_in_transform, target_channel, "ofa_%d" % (i + 1))
            ofa_loss = mx.sym.mean(mx.sym.square(student_feat - teacher_feat))
            ofa_loss = mx.sym.MakeLoss(ofa_loss, grad_scale=grad_scale, name="ofa_loss_%d" % (i + 1))
            ofa_loss_list.append(ofa_loss)
        return tuple(ofa_loss_list)


class OFAFasterRcnn(TridentFasterRcnn):
    def __init__(self):
        super().__init__()

    @classmethod
    def get_train_symbol(cls, backbone, neck, rpn_head, roi_extractor, bbox_head, ofa_head, num_branch, scaleaware):
        gt_bbox = X.var("gt_bbox")
        im_info = X.var("im_info")
        if scaleaware:
            valid_ranges = X.var("valid_ranges")
        rpn_cls_label = X.var("rpn_cls_label")
        rpn_reg_target = X.var("rpn_reg_target")
        rpn_reg_weight = X.var("rpn_reg_weight")

        im_info = TridentResNetV2Builder.stack_branch_symbols([im_info] * num_branch)
        gt_bbox = TridentResNetV2Builder.stack_branch_symbols([gt_bbox] * num_branch)
        if scaleaware:
            valid_ranges = X.reshape(valid_ranges, (-3, -2))
        rpn_cls_label = X.reshape(rpn_cls_label, (-3, -2))
        rpn_reg_target = X.reshape(rpn_reg_target, (-3, -2))
        rpn_reg_weight = X.reshape(rpn_reg_weight, (-3, -2))

        rpn_feat = backbone.get_rpn_feature()
        rcnn_feat = backbone.get_rcnn_feature()
        rpn_feat = neck.get_rpn_feature(rpn_feat)
        rcnn_feat = neck.get_rcnn_feature(rcnn_feat)

        rpn_loss = rpn_head.get_loss(rpn_feat, rpn_cls_label, rpn_reg_target, rpn_reg_weight)
        if scaleaware:
            proposal, bbox_cls, bbox_target, bbox_weight = rpn_head.get_sampled_proposal_with_filter(rpn_feat, gt_bbox, im_info, valid_ranges)
        else:
            proposal, bbox_cls, bbox_target, bbox_weight = rpn_head.get_sampled_proposal(rpn_feat, gt_bbox, im_info)
        roi_feat = roi_extractor.get_roi_feature(rcnn_feat, proposal)
        bbox_loss = bbox_head.get_loss(roi_feat, bbox_cls, bbox_target, bbox_weight)
        ofa_loss = ofa_head.get_loss(rcnn_feat)

        return X.group(rpn_loss + bbox_loss + ofa_loss)

