import mxnet as mx
import mxnext as X

from symbol.builder import Backbone, Neck


class TridentResNetV1bFPN(Backbone):
    def __init__(self, pBackbone):
        super().__init__(pBackbone)
        from mxnext.backbone.resnet_v1b_helper import depth_config, resnet_c1, \
            resnet_c2, resnet_c3, resnet_c4
        from models.tridentnet.resnet_v1b_helper import trident_resnet_c5
        p = self.p
        depth = p.depth
        fp16 = p.fp16
        norm = p.normalizer
        num_trident_block = p.num_trident_block
        branch_dilates = p.branch_dilates
        branch_ids = p.branch_ids
        share_bn = p.share_bn
        share_conv = p.share_conv

        num_c2_unit, num_c3_unit, num_c4_unit, num_c5_unit = depth_config[depth]

        data = X.var("data")
        if fp16:
            data = X.to_fp16(data, "data_fp16")
        c1 = resnet_c1(data, norm)
        c2 = resnet_c2(c1, num_c2_unit, 1, 1, norm)
        c3 = resnet_c3(c2, num_c3_unit, 2, 1, norm)
        c4 = resnet_c4(c3, num_c4_unit, 2, 1, norm)
        c5_1, c5_2, c5_3 = trident_resnet_c5(c4, num_c5_unit, 2, branch_dilates, 
            norm, num_trident_block, branch_ids, share_bn, share_conv)

        self.symbol = c2, c3, c4, c5_1, c5_2, c5_3

    def get_rpn_feature(self):
        return self.symbol

    def get_rcnn_feature(self):
        return self.symbol


class TridentFPNNeck(Neck):
    def __init__(self, pNeck):
        super().__init__(pNeck)
        self.fpn_feat = None

    def add_norm(self, sym):
        p = self.p
        if p.normalizer.__name__ == "fix_bn":
            pass
        elif p.normalizer.__name__ in ["sync_bn", "gn"]:
            sym = p.normalizer(sym)
        else:
            raise NotImplementedError("Unsupported normalizer: {}".format(p.normalizer.__name__))
        return sym

    def fpn_neck(self, data):
        if self.fpn_feat is not None:
            return self.fpn_feat

        c2, c3, c4, c5_1, c5_2, c5_3 = data

        xavier_init = mx.init.Xavier(factor_type="in", rnd_type="uniform", magnitude=3)

        # P5_1
        p5_1 = X.conv(
            data=c5_1,
            filter=256,
            no_bias=False,
            weight=X.var(name="P5_1_lateral_weight", init=xavier_init),
            bias=X.var(name="P5_1_lateral_bias", init=X.zero_init()),
            name="P5_1_lateral"
        )
        p5_1 = self.add_norm(p5_1)
        p5_1_conv = X.conv(
            data=p5_1,
            kernel=3,
            filter=256,
            no_bias=False,
            weight=X.var(name="P5_1_conv_weight", init=xavier_init),
            bias=X.var(name="P5_1_conv_bias", init=X.zero_init()),
            name="P5_1_conv"
        )
        p5_1_conv = self.add_norm(p5_1_conv)

        # P5_2
        p5_2 = X.conv(
            data=c5_2,
            filter=256,
            no_bias=False,
            weight=X.var(name="P5_2_lateral_weight", init=xavier_init),
            bias=X.var(name="P5_2_lateral_bias", init=X.zero_init()),
            name="P5_2_lateral"
        )
        p5_2 = self.add_norm(p5_2)
        p5_2_conv = X.conv(
            data=p5_2,
            kernel=3,
            filter=256,
            no_bias=False,
            weight=X.var(name="P5_2_conv_weight", init=xavier_init),
            bias=X.var(name="P5_2_conv_bias", init=X.zero_init()),
            name="P5_2_conv"
        )
        p5_2_conv = self.add_norm(p5_2_conv)

        # P5_3
        p5_3 = X.conv(
            data=c5_3,
            filter=256,
            no_bias=False,
            weight=X.var(name="P5_3_lateral_weight", init=xavier_init),
            bias=X.var(name="P5_3_lateral_bias", init=X.zero_init()),
            name="P5_3_lateral"
        )
        p5_3 = self.add_norm(p5_3)
        p5_3_conv = X.conv(
            data=p5_3,
            kernel=3,
            filter=256,
            no_bias=False,
            weight=X.var(name="P5_3_conv_weight", init=xavier_init),
            bias=X.var(name="P5_3_conv_bias", init=X.zero_init()),
            name="P5_3_conv"
        )
        p5_3_conv = self.add_norm(p5_3_conv)

        # P4
        p5_up = mx.sym.UpSampling(
            p5_1,
            scale=2,
            sample_type="nearest",
            name="P5_1_upsampling",
            num_args=1
        )
        p4_la = X.conv(
            data=c4,
            filter=256,
            no_bias=False,
            weight=X.var(name="P4_lateral_weight", init=xavier_init),
            bias=X.var(name="P4_lateral_bias", init=X.zero_init()),
            name="P4_lateral"
        )
        p4_la = self.add_norm(p4_la)
        p5_clip = mx.sym.slice_like(p5_up, p4_la, name="P4_clip")
        p4 = mx.sym.add_n(p5_clip, p4_la, name="P4_sum")

        p4_conv = X.conv(
            data=p4,
            kernel=3,
            filter=256,
            no_bias=False,
            weight=X.var(name="P4_conv_weight", init=xavier_init),
            bias=X.var(name="P4_conv_bias", init=X.zero_init()),
            name="P4_conv"
        )
        p4_conv = self.add_norm(p4_conv)

        # P3
        p4_up = mx.sym.UpSampling(
            p4,
            scale=2,
            sample_type="nearest",
            name="P4_upsampling",
            num_args=1
        )
        p3_la = X.conv(
            data=c3,
            filter=256,
            no_bias=False,
            weight=X.var(name="P3_lateral_weight", init=xavier_init),
            bias=X.var(name="P3_lateral_bias", init=X.zero_init()),
            name="P3_lateral"
        )
        p3_la = self.add_norm(p3_la)
        p4_clip = mx.sym.slice_like(p4_up, p3_la, name="P3_clip")
        p3 = mx.sym.add_n(p4_clip, p3_la, name="P3_sum")

        p3_conv = X.conv(
            data=p3,
            kernel=3,
            filter=256,
            no_bias=False,
            weight=X.var(name="P3_conv_weight", init=xavier_init),
            bias=X.var(name="P3_conv_bias", init=X.zero_init()),
            name="P3_conv"
        )
        p3_conv = self.add_norm(p3_conv)

        # P2
        p3_up = mx.sym.UpSampling(
            p3,
            scale=2,
            sample_type="nearest",
            name="P3_upsampling",
            num_args=1
        )
        p2_la = X.conv(
            data=c2,
            filter=256,
            no_bias=False,
            weight=X.var(name="P2_lateral_weight", init=xavier_init),
            bias=X.var(name="P2_lateral_bias", init=X.zero_init()),
            name="P2_lateral"
        )
        p2_la = self.add_norm(p2_la)
        p3_clip = mx.sym.slice_like(p3_up, p2_la, name="P2_clip")
        p2 = mx.sym.add_n(p3_clip, p2_la, name="P2_sum")

        p2_conv = X.conv(
            data=p2,
            kernel=3,
            filter=256,
            no_bias=False,
            weight=X.var(name="P2_conv_weight", init=xavier_init),
            bias=X.var(name="P2_conv_bias", init=X.zero_init()),
            name="P2_conv"
        )
        p2_conv = self.add_norm(p2_conv)

        # P6
        p6 = X.max_pool(
            p5_3_conv,
            name="P6_subsampling",
            kernel=1,
            stride=2,
        )

        conv_fpn_feat = dict(
            stride64=p6,
            stride32=p5_2_conv,
            stride16=p4_conv,
            stride8=p3_conv,
            stride4=p2_conv
        )

        self.fpn_feat = conv_fpn_feat
        return self.fpn_feat

    def get_rpn_feature(self, rpn_feat):
        return self.fpn_neck(rpn_feat)

    def get_rcnn_feature(self, rcnn_feat):
        return self.fpn_neck(rcnn_feat)
