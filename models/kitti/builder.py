import mxnet as mx
import mxnext as X
from mxnext import conv, relu, add, sigmoid
from mxnext.backbone.resnet_v1b_helper import resnet_unit
from symbol.builder import Backbone
from models.efficientnet.builder import se
from models.dcn.builder import hybrid_resnet_fpn_builder
from models.maskrcnn.builder import MaskFasterRcnnHead


def sp_resnet_v1b_unit(input, name, filter, stride, dilate, proj, norm, **kwargs):
    conv1 = conv(input, name=name + "_conv1", filter=filter // 4)
    bn1 = norm(conv1, name=name + "_bn1")
    relu1 = relu(bn1, name=name + "_relu1")

    relu1u, relu1d = mx.sym.split(relu1, num_outputs=2, axis=2)
    relu1du, relu1dd = mx.sym.split(relu1d, num_outputs=2, axis=2)

    conv2_weight = X.var(name + "_conv2_weight")
    conv2_params = dict(
        weight=conv2_weight,
        stride=stride,
        filter=filter // 4,
        kernel=3)
    conv2u = conv(relu1u, name=name + "_conv2u", dilate=1, **conv2_params)
    conv2du = conv(relu1du, name=name + "_conv2du", dilate=2, **conv2_params)
    conv2dd = conv(relu1dd, name=name + "_conv2dd", dilate=3, **conv2_params)
    conv2 = X.concat([conv2u, conv2du, conv2dd], axis=2, name=name + "_conv2")
    bn2 = norm(conv2, name=name + "_bn2")
    relu2 = relu(bn2, name=name + "_relu2")

    conv3 = conv(relu2, name=name + "_conv3", filter=filter)
    bn3 = norm(conv3, name=name + "_bn3")

    if proj:
        shortcut = conv(input, name=name + "_sc", filter=filter, stride=stride)
        shortcut = norm(shortcut, name=name + "_sc_bn")
    else:
        shortcut = input

    eltwise = add(bn3, shortcut, name=name + "_plus")

    return relu(eltwise, name=name + "_relu")


def spv2_resnet_v1b_unit(input, name, filter, stride, dilate, proj, norm, **kwargs):
    conv1 = conv(input, name=name + "_conv1", filter=filter // 4)
    bn1 = norm(conv1, name=name + "_bn1")
    relu1 = relu(bn1, name=name + "_relu1")

    relu1u, relu1d = mx.sym.split(relu1, num_outputs=2, axis=2)

    conv2_weight = X.var(name + "_conv2_weight")
    conv2_params = dict(
        weight=conv2_weight,
        stride=stride,
        filter=filter // 4,
        kernel=3)
    conv2u = conv(relu1u, name=name + "_conv2u", dilate=1, **conv2_params)
    conv2d = conv(relu1d, name=name + "_conv2d", dilate=2, **conv2_params)
    conv2 = X.concat([conv2u, conv2d], axis=2, name=name + "_conv2")
    bn2 = norm(conv2, name=name + "_bn2")
    relu2 = relu(bn2, name=name + "_relu2")

    conv3 = conv(relu2, name=name + "_conv3", filter=filter)
    bn3 = norm(conv3, name=name + "_bn3")

    if proj:
        shortcut = conv(input, name=name + "_sc", filter=filter, stride=stride)
        shortcut = norm(shortcut, name=name + "_sc_bn")
    else:
        shortcut = input

    eltwise = add(bn3, shortcut, name=name + "_plus")

    return relu(eltwise, name=name + "_relu")


def dilated_resnet_v1b_unit(input, name, filter, stride, dilate, proj, norm, **kwargs):
    conv1 = conv(input, name=name + "_conv1", filter=filter // 4)
    bn1 = norm(conv1, name=name + "_bn1")
    relu1 = relu(bn1, name=name + "_relu1")

    conv2 = conv(relu1, name=name + "_conv2", dilate=2, stride=stride, filter=filter // 4, kernel=3)
    bn2 = norm(conv2, name=name + "_bn2")
    relu2 = relu(bn2, name=name + "_relu2")

    conv3 = conv(relu2, name=name + "_conv3", filter=filter)
    bn3 = norm(conv3, name=name + "_bn3")

    if proj:
        shortcut = conv(input, name=name + "_sc", filter=filter, stride=stride)
        shortcut = norm(shortcut, name=name + "_sc_bn")
    else:
        shortcut = input

    eltwise = add(bn3, shortcut, name=name + "_plus")

    return relu(eltwise, name=name + "_relu")


SpResNetV1bFPN = hybrid_resnet_fpn_builder(sp_resnet_v1b_unit)
Spv2ResNetV1bFPN = hybrid_resnet_fpn_builder(spv2_resnet_v1b_unit)
DilatedResNetV1bFPN = hybrid_resnet_fpn_builder(dilated_resnet_v1b_unit)