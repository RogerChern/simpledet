import mxnet as mx
import mxnext as X
from mxnext import conv, relu, add
from mxnext.backbone.resnet_v1b_helper import resnet_unit
from symbol.builder import Backbone, Neck


class DilatedResNetV1bC4(Backbone):
    def __init__(self, pBackbone):
        super().__init__(pBackbone)
        p = self.p

        import mxnext.backbone.resnet_v1b_helper as helper
        num_c2, num_c3, num_c4, _ = helper.depth_config[p.depth]

        data = X.var("data")
        if p.fp16:
            data = data.astype("float16")
        c1 = helper.resnet_c1(data, p.normalizer)
        c2 = helper.resnet_c2(c1, num_c2, 1, p.c2_dil or 1, p.normalizer)
        c3 = helper.resnet_c3(c2, num_c3, 2, p.c3_dil or 1, p.normalizer)
        c4 = helper.resnet_c4(c3, num_c4, 2, p.c4_dil or 1, p.normalizer)

        self.symbol = c4

    def get_rpn_feature(self):
        return self.symbol

    def get_rcnn_feature(self):
        return self.symbol


class DilatedResNetV1bC5(Backbone):
    def __init__(self, pBackbone):
        super().__init__(pBackbone)
        p = self.p

        import mxnext.backbone.resnet_v1b_helper as helper
        num_c2, num_c3, num_c4, num_c5 = helper.depth_config[p.depth]

        data = X.var("data")
        if p.fp16:
            data = data.astype("float16")
        c1 = helper.resnet_c1(data, p.normalizer)
        c2 = helper.resnet_c2(c1, num_c2, 1, p.c2_dil or 1, p.normalizer)
        c3 = helper.resnet_c3(c2, num_c3, 2, p.c3_dil or 1, p.normalizer)
        c4 = helper.resnet_c4(c3, num_c4, 2, p.c4_dil or 1, p.normalizer)
        c5 = helper.resnet_c5(c4, num_c5, 1, p.c5_dil or 1, p.normalizer)

        self.c4 = c4
        self.c5 = c5

    def get_rpn_feature(self):
        return self.c4

    def get_rcnn_feature(self):
        return self.c5


class DilatedResNetV1bFPN(Backbone):
    def __init__(self, pBackbone):
        super().__init__(pBackbone)
        p = self.p

        import mxnext.backbone.resnet_v1b_helper as helper
        num_c2, num_c3, num_c4, num_c5 = helper.depth_config[p.depth]

        data = X.var("data")
        if p.fp16:
            data = data.astype("float16")
        c1 = helper.resnet_c1(data, p.normalizer)
        c2 = helper.resnet_c2(c1, num_c2, 1, p.c2_dil or 1, p.normalizer)
        c3 = helper.resnet_c3(c2, num_c3, 2, p.c3_dil or 1, p.normalizer)
        c4 = helper.resnet_c4(c3, num_c4, 2, p.c4_dil or 1, p.normalizer)
        c5 = helper.resnet_c5(c4, num_c5, 2, p.c5_dil or 1, p.normalizer)

        self.symbol = (c2, c3, c4, c5)

    def get_rpn_feature(self):
        return self.symbol

    def get_rcnn_feature(self):
        return self.symbol


class ASPP(Neck):
    def __init__(self, pNeck):
        super().__init__(pNeck)

    def _aspp(self, feat, prefix):
        p = self.p

        # transform
        use_relu_for_transform_conv = p.use_relu_for_transform_conv or False
        trans_conv = X.convrelu if use_relu_for_transform_conv else X.conv
        GAP = X.global_avg_pool(feat, name="%s_aspp_gap" % prefix)
        GAP = X.broadcast_like(GAP, feat, name="%s_aspp_gap_broadcast" % prefix)
        conv_1x1 = trans_conv(feat, "%s_aspp_conv_1x1" % prefix, 256)
        conv_3x3_dil6 = trans_conv(feat, "%s_aspp_conv_3x3_dil6" % prefix, 256, 3, dilate=6)
        conv_3x3_dil12 = trans_conv(feat, "%s_aspp_conv_3x3_dil12" % prefix, 256, 3, dilate=12)
        conv_3x3_dil18 = trans_conv(feat, "%s_aspp_conv_3x3_dil18" % prefix, 256, 3, dilate=18)
        concated = X.concat([GAP, conv_1x1, conv_3x3_dil6, conv_3x3_dil12, conv_3x3_dil18], name="%s_aspp_concatd" % prefix)

        # output
        use_relu_for_output_conv = p.use_relu_for_output_conv or False
        out_conv = X.convrelu if use_relu_for_output_conv else X.conv
        output = out_conv(concated, "%s_aspp_output" % prefix, 256)
        
        return output

    def get_rpn_feature(self, rpn_feat):
        rpn_feat = self._aspp(rpn_feat, "rpn")
        return rpn_feat

    def get_rcnn_feature(self, rcnn_feat):
        rcnn_feat = self._aspp(rcnn_feat, "rcnn")
        return rcnn_feat

