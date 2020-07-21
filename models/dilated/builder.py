import mxnet as mx
import mxnext as X
from mxnext import conv, relu, add
from mxnext.backbone.resnet_v1b_helper import resnet_unit
from symbol.builder import Backbone


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
