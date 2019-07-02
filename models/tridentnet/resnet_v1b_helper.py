import mxnet as mx
import mxnext as X

from mxnext.backbone.resnet_v1b_helper import resnet_unit


def bn_p(data, name, norm, branch_ids, share_weight=True):
    gamma = X.var(name + "_gamma")
    beta = X.var(name + "_beta")
    moving_mean = X.var(name + "_moving_mean")
    moving_var = X.var(name + "_moving_var")

    bn_layers = []
    for i, data_i in zip(branch_ids, data):
        if share_weight:
            bn_i = norm(data=data_i, name=name + "_shared%d" % i, gamma=gamma, 
                beta=beta, moving_mean=moving_mean, moving_var=moving_var)
        else:
            bn_i = norm(data=data_i, name=name + "_branch%d" % i)
        bn_layers.append(bn_i)
    return bn_layers

def conv_p(data, name, kernel, filter, branch_ids, no_bias=True, 
    share_weight=True, pad=(0, 0), stride=(1, 1), dilate=(1, 1)):

    weight = X.var(name + '_weight')
    bias = None if no_bias else X.var(name + '_bias')

    conv_layers = []
    for i in range(len(data)):
        data_i = data[i]
        stride_i = stride[i] if type(stride) is list else stride
        dilate_i = dilate[i] if type(dilate) is list else dilate
        pad_i = pad[i] if type(pad) is list else pad
        branch_i = branch_ids[i]
        if share_weight:
            conv_i = X.conv(data=data_i, kernel=kernel, filter=filter, stride=stride_i, 
                dilate=dilate_i, pad=pad_i, name=name + '_shared%d' % branch_i, no_bias=no_bias, 
                weight=weight, bias=bias)
        else:
            conv_i = X.conv(data=data_i, kernel=kernel, filter=filter, stride=stride_i, 
                dilate=dilate_i, pad=pad_i, name=name + '_branch%d' % branch_i, no_bias=no_bias)
        conv_layers.append(conv_i)
    return conv_layers

def relu_p(data, name, branch_ids):
    return [X.relu(d, name=name + "_branch%d" % i) for i, d in zip(branch_ids, data)]

def stack_branch_symbols(data_list):
    data = mx.symbol.stack(*data_list, axis=1)
    data = mx.symbol.Reshape(data, (-3, -2))
    return data

def trident_resnet_unit(data, name, filter, stride, dilate, proj, norm, branch_ids, 
    share_bn, share_conv):

    conv1 = conv_p(data, name=name + "_conv1", filter=filter // 4, 
        kernel=1, branch_ids=branch_ids, share_weight=share_conv)
    bn1 = bn_p(conv1, name=name + "_bn1", norm=norm, 
        branch_ids=branch_ids, share_weight=share_bn)
    relu1 = relu_p(bn1, name=name + "_relu1", branch_ids=branch_ids)

    conv2 = conv_p(relu1, name=name + "_conv2", filter=filter // 4, 
        kernel=3, pad=dilate, stride=stride, dilate=dilate,
        branch_ids=branch_ids, share_weight=share_conv)
    bn2 = bn_p(conv2, name=name + "_bn2", norm=norm, 
        branch_ids=branch_ids, share_weight=share_bn)
    relu2 = relu_p(bn2, name=name + "_relu2", branch_ids=branch_ids)

    conv3 = conv_p(relu2, name=name + "_conv3", filter=filter, kernel=1,
        branch_ids=branch_ids, share_weight=share_conv)
    bn3 = bn_p(conv3, name=name + "_bn3", norm=norm, 
        branch_ids=branch_ids, share_weight=share_bn)

    if proj:
        shortcut = conv_p(data, name=name + "_sc", filter=filter, kernel=1, 
            stride=stride, branch_ids=branch_ids, share_weight=share_conv)
        shortcut = bn_p(shortcut, name=name + "_sc_bn", norm=norm, 
            branch_ids=branch_ids, share_weight=share_bn)
    else:
        shortcut = data

    plus = [X.add(bn3_i, shortcut_i, name=name + "_plus_branch%d" % i) \
            for i, bn3_i, shortcut_i in zip(branch_ids, bn3, shortcut)]
    return relu_p(plus, name=name + "_relu3", branch_ids=branch_ids)

def trident_resnet_stage(data, name, num_block, filter, stride, dilate, proj,
    norm, num_trident_block, branch_ids, share_bn, share_conv, **kwargs):
    num_trident_block = num_trident_block or num_block  # transform all blocks by default

    for i in range(1, num_block + 1):
        s = stride if i == 1 else 1
        d = dilate
        p = proj if i == 1 else False

        # [i ... num_block] == [1 ... num_trident_block]
        if i == (num_block - num_trident_block + 1):
            data = [data] * len(branch_ids)
        if i >= (num_block - num_trident_block + 1):
            data = trident_resnet_unit(data, "{}_unit{}".format(name, i), filter, s, d, 
                p, norm, branch_ids, share_bn, share_conv)
        else:
            data = resnet_unit(data, "{}_unit{}".format(name, i), filter, s, 1, p, norm)
    return data

def trident_resnet_c4(data, num_block, stride, dilate, norm, num_trident_block, 
    branch_ids, share_bn, share_conv):
    return trident_resnet_stage(data, "stage3", num_block, 1024, stride, dilate, True,
        norm, num_trident_block, branch_ids, share_bn, share_conv)

def trident_resnet_c5(data, num_block, stride, dilate, norm, num_trident_block, 
    branch_ids, share_bn, share_conv):
    return trident_resnet_stage(data, "stage4", num_block, 2048, stride, dilate, True,
        norm, num_trident_block, branch_ids, share_bn, share_conv)

