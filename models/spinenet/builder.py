from functools import partial

import mxnet as mx
import mxnext as X
from mxnext import conv, relu, add, sigmoid, convnormrelu, max_pool
from symbol.builder import Backbone, Neck


def residual_unit(input, name, filter, norm, **kwargs):
    conv1 = conv(input, name=name + "_conv1", filter=filter, kernel=3)
    bn1 = norm(conv1, name=name + "_bn1")
    relu1 = relu(bn1, name=name + "_relu1")

    conv2 = conv(relu1, name=name + "_conv2", filter=filter, kernel=3)
    bn2 = norm(conv2, name=name + "_bn2")

    eltwise = add(bn2, input, name=name + "_plus")

    return relu(eltwise, name=name + "_relu")


def bottleneck_unit(input, name, filter, norm, proj, **kwargs):
    conv1 = conv(input, name=name + "_conv1", filter=filter)
    bn1 = norm(conv1, name=name + "_bn1")
    relu1 = relu(bn1, name=name + "_relu1")

    conv2 = conv(relu1, name=name + "_conv2", filter=filter, kernel=3)
    bn2 = norm(conv2, name=name + "_bn2")
    relu2 = relu(bn2, name=name + "_relu2")

    conv3 = conv(relu2, name=name + "_conv3", filter=filter * 4)
    bn3 = norm(conv3, name=name + "_bn3")

    if proj:
        shortcut = conv(input, name=name + "_sc", filter=filter * 4)
        shortcut = norm(shortcut, name=name + "_sc_bn")
    else:
        shortcut = input

    eltwise = add(bn3, shortcut, name=name + "_plus")

    return relu(eltwise, name=name + "_relu")


def nn_upsample(x, scale, name):
    return mx.sym.UpSampling(x, scale=scale, name=name, sample_type="nearest")


def elemwise_add(x, y, name):
    return mx.sym.elemwise_add(x, y, name=name)


def identity(input, name, norm, ifilter, ofilter, alpha, ratio, **kwargs):
    assert ratio == 1
    conv1 = convnormrelu(norm, input, name=name + "_id_conv1", filter=int(ifilter * alpha))
    conv2 = convnormrelu(norm, conv1, name=name + "_id_conv2", filter=ofilter)
    return conv2


def upsample(input, name, norm, ifilter, ofilter, alpha, ratio, **kwargs):
    conv1 = convnormrelu(norm, input, name=name + "_up_conv1", filter=int(ifilter * alpha))
    up = nn_upsample(conv1, scale=ratio, name=name + "_up_nnupsample%d" % ratio)
    conv2 = convnormrelu(norm, up, name=name + "_up_conv2", filter=ofilter)
    return conv2


def downsample(input, name, norm, ifilter, ofilter, alpha, ratio, **kwargs):
    conv1 = convnormrelu(norm, input, name=name + "_down_conv1", filter=int(ifilter * alpha))
    down = convnormrelu(norm, conv1, name=name + "_down_stride2", filter=int(ifilter * alpha), stride=2)
    max_pool_ratio = ratio // 2
    if max_pool_ratio != 1:
        down = max_pool(down, name=name + "_down_maxpool%d" % max_pool_ratio, kernel=max_pool_ratio, stride=max_pool_ratio, pad=0)
    conv2 = convnormrelu(norm, down, name=name + "_down_conv2", filter=ofilter)
    return conv2


def spinenet(data, norm, alpha=0.5, width=1.0, repeat=1):
    r256 = partial(residual_unit, norm=norm, filter=int(256 * width))
    b64 = partial(bottleneck_unit, norm=norm, filter=int(64 * width), proj=True)
    b128 = partial(bottleneck_unit, norm=norm, filter=int(128 * width), proj=True)
    b256 = partial(bottleneck_unit, norm=norm, filter=int(256 * width), proj=True)
    u2 = partial(upsample, norm=norm, ratio=2, alpha=alpha)
    u4 = partial(upsample, norm=norm, ratio=4, alpha=alpha)
    d2 = partial(downsample, norm=norm, ratio=2, alpha=alpha)
    d4 = partial(downsample, norm=norm, ratio=4, alpha=alpha)
    d8 = partial(downsample, norm=norm, ratio=8, alpha=alpha)
    id = partial(identity, norm=norm, ratio=1, alpha=alpha)

    def resize(input, name, istride, ostride, ifilter, ofilter, **kwargs):
        # pow of 2 is exact in float
        idivo = {
            0.25:  u4,
            0.5:   u2,
            1.0:   id,
            2.0:   d2,
            4.0:   d4,
            8.0:   d8
        }
        return idivo[ostride / istride](input, name=name, ifilter=ifilter, ofilter=ofilter)

    specs = [
        [],
        #block type,  ichannel, ochannel, stride,   inputs...
        [b64,         64,       256,      4,        data],
        [b64,         64,       256,      4,        1],
        [b64,         64,       256,      4,        1,       2],
        [r256,        256,      256,      16,       2,       1],
        [b128,        128,      512,      8,        3,       4],
        [b256,        256,      1024,     16,       5,       3],
        [r256,        256,      256,      64,       4,       6],
        [b256,        256,      1024,     16,       6,       4],
        [r256,        256,      256,      32,       7,       8],
        [r256,        256,      256,      128,      9,       7],
        [b256,        256,      1024,     32,       9,       10],
        [b256,        256,      1024,     32,       11,      9],
        [b256,        256,      1024,     16,       6,       11],
        [b128,        128,      512,      8,        11,      5],
        [b256,        256,      1024,     32,       8,       12,      13],
        [b256,        256,      1024,     128,      6,       15],
        [b256,        256,      1024,     64,       15,      13]
    ]
    # modulated by width factor
    for spec in specs[1:]:
        spec[1] = int(spec[1] * width)
        spec[2] = int(spec[2] * width)

    blocks = [data]
    for i, spec in enumerate(specs[1:], start=1):
        op, target_ichannel, target_ochannel, target_stride, *inputs = spec
        if len(inputs) == 1:
            if type(inputs[0]) != int:
                # block1
                parent_sym = inputs[0]
                for j in range(1, repeat + 1):
                    parent_sym = op(parent_sym, name="block%d_repeat%d" % (i, j))
                blocks.append(parent_sym)
            else:
                # block2
                parent_sym = blocks[inputs[0]]
                for j in range(1, repeat + 1):
                    parent_sym = op(parent_sym, name="block%d_repeat%d" % (i, j))
                blocks.append(parent_sym)
        else:
            parent_syms = []
            for j, idx in enumerate(inputs, start=1):
                parent_block = blocks[idx]
                parent_ichannel = specs[idx][1]
                parent_ochannel = specs[idx][2]
                parent_stride = specs[idx][3]
                parent_sym = resize(parent_block, "block%d_resize%d" % (i, j), parent_stride, target_stride, parent_ochannel, target_ichannel)
                parent_syms.append(parent_sym)
            parent_sym = X.add_n(*parent_syms, name="block%d_addn" % i)
            for j in range(1, repeat + 1):
                parent_sym = op(parent_sym, name="block%d_repeat%d" % (i, j))
            blocks.append(parent_sym)
    return blocks[-5:]


def spinenet_builder(channel_mult, block_repeat, alpha, neck_channel):
    class SpineNet(Backbone):
        def __init__(self, pBackbone):
            super().__init__(pBackbone)
            p = self.p
            norm = p.normalizer

            from mxnext.backbone.resnet_v1b_helper import resnet_c1
            data = X.var("data")
            if p.fp16:
                data = X.to_fp16(data, name="data_fp16")
            c1 = resnet_c1(data, norm)
            c4, c3, c5, c7, c6 = spinenet(c1, norm, width=channel_mult, repeat=block_repeat, alpha=alpha)
            self.symbol = (c3, c4, c5, c6, c7)
            self._attatch_1x1()

        def _attatch_1x1(self):
            c3, c4, c5, c6, c7 = self.symbol
            p3 = X.convrelu(c3, name="p3", filter=neck_channel)
            p4 = X.convrelu(c4, name="p4", filter=neck_channel)
            p5 = X.convrelu(c5, name="p5", filter=neck_channel)
            p6 = X.convrelu(c6, name="p6", filter=neck_channel)
            p7 = X.convrelu(c7, name="p7", filter=neck_channel)
            self.symbol = dict(
                stride8=p3,
                stride16=p4,
                stride32=p5,
                stride64=p6,
                stride128=p7
            )

        def get_rpn_feature(self):
            return self.symbol

        def get_rcnn_feature(self):
            return self.symbol
    return SpineNet


SpineNet49s = spinenet_builder(channel_mult=0.75, block_repeat=1, alpha=0.5, neck_channel=128)
SpineNet49 = spinenet_builder(channel_mult=1.0, block_repeat=1, alpha=0.5, neck_channel=256)
SpineNet98 = spinenet_builder(channel_mult=1.0, block_repeat=2, alpha=0.5, neck_channel=256)
SpineNet147 = spinenet_builder(channel_mult=1.0, block_repeat=3, alpha=1.0, neck_channel=256)


def test_spinenet49s():
    class BackboneParam:
        normalizer = X.normalizer_factory()
        width = 1.0

    net = SpineNet49s(BackboneParam)
    sym = net.get_rpn_feature()
    sym = X.group(sym)
    sym.save("spinenet49s.json")
    _, out_shape, _ = sym.get_internals().infer_shape(data=(1, 3, 640, 640))
    out_shape_dict = dict(zip(sym.get_internals().list_outputs(), out_shape))

    assert out_shape_dict["block1_repeat1_relu_output"][-3:] == (192, 160, 160)
    assert out_shape_dict["block2_repeat1_relu_output"][-3:] == (192, 160, 160)
    assert out_shape_dict["block3_repeat1_relu_output"][-3:] == (192, 160, 160)
    assert out_shape_dict["block4_repeat1_relu_output"][-3:] == (192, 40, 40)
    assert out_shape_dict["block5_repeat1_relu_output"][-3:] == (384, 80, 80)
    assert out_shape_dict["block6_repeat1_relu_output"][-3:] == (768, 40, 40)
    assert out_shape_dict["block7_repeat1_relu_output"][-3:] == (192, 10, 10)
    assert out_shape_dict["block8_repeat1_relu_output"][-3:] == (768, 40, 40)
    assert out_shape_dict["block9_repeat1_relu_output"][-3:] == (192, 20, 20)
    assert out_shape_dict["block10_repeat1_relu_output"][-3:] == (192, 5, 5)
    assert out_shape_dict["block11_repeat1_relu_output"][-3:] == (768, 20, 20)
    assert out_shape_dict["block12_repeat1_relu_output"][-3:] == (768, 20, 20)
    assert out_shape_dict["block13_repeat1_relu_output"][-3:] == (768, 40, 40)
    assert out_shape_dict["block14_repeat1_relu_output"][-3:] == (384, 80, 80)
    assert out_shape_dict["block15_repeat1_relu_output"][-3:] == (768, 20, 20)
    assert out_shape_dict["block16_repeat1_relu_output"][-3:] == (768, 5, 5)
    assert out_shape_dict["block17_repeat1_relu_output"][-3:] == (768, 10, 10)
    assert out_shape_dict["p3_relu_output"][-3:] == (128, 80, 80)
    assert out_shape_dict["p4_relu_output"][-3:] == (128, 40, 40)
    assert out_shape_dict["p5_relu_output"][-3:] == (128, 20, 20)
    assert out_shape_dict["p6_relu_output"][-3:] == (128, 10, 10)
    assert out_shape_dict["p7_relu_output"][-3:] == (128, 5, 5)

    return sym


if __name__ == "__main__":
    sym = test_spinenet49s()
    mx.viz.print_summary(sym, shape=dict(data=(1, 3, 640, 640)))
