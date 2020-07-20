import mxnet as mx
import mxnext as X
from mxnext import conv, relu, add
from mxnext.backbone import resnet_v1_helper, resnet_v1b_helper
from symbol.builder import Backbone


def trident_resnet_v1_unit(input, name, id, filter, stride, dilate, proj, **kwargs):
    p = kwargs["params"]
    share_bn = p.branch_bn_shared
    share_conv = p.branch_conv_shared
    norm = p.normalizer

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

    bn_params = lambda x: dict(
        gamma=X.shared_var(name + "_%s_gamma" % x) if share_bn else None,
        beta=X.shared_var(name + "_%s_beta" % x) if share_bn else None,
        moving_mean=X.shared_var(name + "_%s_moving_mean" % x) if share_bn else None,
        moving_var=X.shared_var(name + "_%s_moving_var" % x) if share_bn else None,
        name=name + "_%s" % x + bn_postfix
    )

    ######################### construct graph #########################
    conv1 = conv(input, filter=filter // 4, stride=stride, **conv_params("conv1"))
    bn1 = norm(conv1, **bn_params("bn1"))
    relu1 = relu(bn1, name=name + other_postfix)

    conv2 = conv(relu1, filter=filter // 4, kernel=3, dilate=dilate, **conv_params("conv2"))
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


def trident_resnet_v1b_unit(input, name, id, filter, stride, dilate, proj, **kwargs):
    """
    Compared with v1, v1b moves stride=2 to the 3x3 conv instead of the 1x1 conv and use std in pre-processing
    This is also known as the facebook re-implementation of ResNet(a.k.a. the torch ResNet)
    """
    p = kwargs["params"]
    share_bn = p.branch_bn_shared
    share_conv = p.branch_conv_shared
    norm = p.normalizer

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


def get_trident_resnet_c4_backbone(unit, helper):
    def build_trident_stage(input, num_block, num_tri, filter, stride, prefix, p):
        # construct leading res blocks
        data = input
        for i in range(1, num_block - num_tri + 1):
            data = unit(
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
        for dil, id in zip(p.branch_dilates, p.branch_ids):
            c = data  # reset c to the output of last stage
            for i in range(num_block - num_tri + 1, num_block + 1):
                if p.branch_deform and i >= num_block - 2:
                    # convert last 3 blocks into deformable conv
                    c = trident_resnet_v1b_deform_unit(
                        input=c,
                        name="%s_unit%s" % (prefix, i),
                        id=id,
                        filter=filter,
                        stride=stride if i == 1 else 1,
                        proj=True if i == 1 else False,
                        dilate=dil,
                        params=p)
                else:
                    c = trident_resnet_v1b_unit(
                        input=c,
                        name="%s_unit%s" % (prefix, i),
                        id=id,
                        filter=filter,
                        stride=stride if i == 1 else 1,
                        proj=True if i == 1 else False,
                        dilate=dil,
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
                c2 = build_trident_stage(c1, num_c2, num_tri, 256, 1, "stage1", p)
                c3 = helper.resnet_c3(c2, num_c3, 2, 1, p.normalizer)
                c4 = helper.resnet_c4(c3, num_c4, 2, 1, p.normalizer)
            elif branch_stage == 3:
                c2 = helper.resnet_c2(c1, num_c2, 1, 1, p.normalizer)
                c3 = build_trident_stage(c2, num_c3, num_tri, 512, 2, "stage2", p)
                c4 = helper.resnet_c4(c3, num_c4, 2, 1, p.normalizer)
            elif branch_stage == 4:
                c2 = helper.resnet_c2(c1, num_c2, 1, 1, p.normalizer)
                c3 = helper.resnet_c3(c2, num_c3, 2, 1, p.normalizer)
                c4 = build_trident_stage(c3, num_c4, num_tri, 1024, 2, "stage3", p)
            else:
                raise ValueError("Unknown branch stage: %d" % branch_stage)

            self.symbol = c4

        def get_rpn_feature(self):
            return self.symbol

        def get_rcnn_feature(self):
            return self.symbol

    return TridentResNetC4


def get_trident_resnet_dilatedc5_backbone(unit, helper):
    def build_trident_stage(input, num_block, num_tri, filter, stride, prefix, p):
        # construct leading res blocks
        data = input
        for i in range(1, num_block - num_tri + 1):
            data = unit(
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
        for dil, id in zip(p.branch_dilates, p.branch_ids):
            c = data  # reset c to the output of last stage
            for i in range(num_block - num_tri + 1, num_block + 1):
                if p.branch_deform and i >= num_block - 2:
                    # convert last 3 blocks into deformable conv
                    c = trident_resnet_v1b_deform_unit(
                        input=c,
                        name="%s_unit%s" % (prefix, i),
                        id=id,
                        filter=filter,
                        stride=stride if i == 1 else 1,
                        proj=True if i == 1 else False,
                        dilate=dil,
                        params=p)
                else:
                    c = trident_resnet_v1b_unit(
                        input=c,
                        name="%s_unit%s" % (prefix, i),
                        id=id,
                        filter=filter,
                        stride=stride if i == 1 else 1,
                        proj=True if i == 1 else False,
                        dilate=dil,
                        params=p)
            cs.append(c)
        return cs

    def stack_trident_branches(blocks):
        block = mx.sym.stack(*blocks, axis=1)
        block = mx.sym.reshape(block, shape=(-3, -2))
        return block

    class TridentResNetDilatedC5(Backbone):
        def __init__(self, pBackbone):
            super().__init__(pBackbone)
            p = self.p

            num_c2, num_c3, num_c4, num_c5 = helper.depth_config[p.depth]
            branch_stage = p.branch_stage or 4
            num_tri = eval("p.num_c%d_block" % branch_stage) or (eval("num_c%d" % branch_stage) - 1)

            ################### construct symbolic graph ###################
            data = X.var("data")
            if p.fp16:
                data = data.astype("float16")
            c1 = helper.resnet_c1(data, p.normalizer)

            if branch_stage == 2:
                c2 = build_trident_stage(c1, num_c2, num_tri, 256, 1, "stage1", p)
                c2 = stack_trident_branches(c2)
                c3 = helper.resnet_c3(c2, num_c3, 2, 1, p.normalizer)
                c4 = helper.resnet_c4(c3, num_c4, 2, 1, p.normalizer)
                c5 = helper.resnet_c5(c4, num_c5, 1, 2, p.normalizer)
            elif branch_stage == 3:
                c2 = helper.resnet_c2(c1, num_c2, 1, 1, p.normalizer)
                c3 = build_trident_stage(c2, num_c3, num_tri, 512, 2, "stage2", p)
                c3 = stack_trident_branches(c3)
                c4 = helper.resnet_c4(c3, num_c4, 2, 1, p.normalizer)
                c5 = helper.resnet_c5(c4, num_c5, 1, 2, p.normalizer)
            elif branch_stage == 4:
                c2 = helper.resnet_c2(c1, num_c2, 1, 1, p.normalizer)
                c3 = helper.resnet_c3(c2, num_c3, 2, 1, p.normalizer)
                c4 = build_trident_stage(c3, num_c4, num_tri, 1024, 2, "stage3", p)
                c4 = stack_trident_branches(c4)
                c5 = helper.resnet_stage(c4, "stage4", num_c5, 1024, 1, 2, p.normalizer)
            elif branch_stage == 5:
                c2 = helper.resnet_c2(c1, num_c2, 1, 1, p.normalizer)
                c3 = helper.resnet_c3(c2, num_c3, 2, 1, p.normalizer)
                c4 = helper.resnet_c4(c3, num_c4, 2, 1, p.normalizer)
                c5 = build_trident_stage(c4, num_c5, num_tri, 2048, 1, "stage4", p)
                c5 = stack_trident_branches(c5)
            else:
                raise ValueError("Unknown branch stage: %d" % branch_stage)

            self.symbol = c5

        def get_rpn_feature(self):
            return self.symbol

        def get_rcnn_feature(self):
            return self.symbol

    return TridentResNetDilatedC5


TridentResNetV1C4 = get_trident_resnet_c4_backbone(trident_resnet_v1_unit, resnet_v1_helper)
TridentResNetV1bC4 = get_trident_resnet_c4_backbone(trident_resnet_v1b_unit, resnet_v1b_helper)
TridentResNetV1bDilatedC5 = get_trident_resnet_dilatedc5_backbone(trident_resnet_v1b_unit, resnet_v1b_helper)
