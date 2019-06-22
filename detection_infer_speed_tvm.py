import argparse
import importlib
import os
import time

from utils.load_model import load_checkpoint

import numpy as np
import mxnet as mx
import tvm
import tvm.relay.testing
from tvm import autotvm, relay
from tvm.contrib.debugger import debug_runtime as graph_runtime


def parse_args():
    parser = argparse.ArgumentParser(description='Test detector inference speed')
    # general
    parser.add_argument('--config', help='config file path', type=str, required=True)
    parser.add_argument('--shape', help='specify input 2d image shape', metavar=('SHORT', 'LONG'), type=int, nargs=2, required=True)
    parser.add_argument('--device', help='target device, one of x86 or cuda', type=str, required=True)
    parser.add_argument('--gpu', help='GPU index', type=int, default=0)
    parser.add_argument('--count', help='number of runs, final result will be averaged', type=int, default=100)
    args = parser.parse_args()

    config = importlib.import_module(args.config.replace('.py', '').replace('/', '.'))
    return config, args.device, args.gpu, args.shape, args.count


if __name__ == "__main__":
    config, device, gpu, shape, count = parse_args()

    pGen, pKv, pRpn, pRoi, pBbox, pDataset, pModel, pOpt, pTest, \
        transform, data_name, label_name, metric_list = config.get_config(is_train=False)

    ctx = tvm.gpu(gpu) if device == "cuda" else tvm.cpu()

    sym = pModel.test_symbol

    # create dummy data batch
    data = mx.nd.ones(shape=[1, 3] + shape)
    im_info = mx.nd.array([x / 2.0 for x in shape] + [2.0]).reshape(1, 3)
    im_id = mx.nd.array([1])
    rec_id = mx.nd.array([1])
    data_names = ["data", "im_info", "im_id", "rec_id"]
    inputs = {k: tvm.nd.array(d.asnumpy(), ctx=ctx) for k, d in zip(data_names, [data, im_info, im_id, rec_id])}

    arg_params, aux_params = load_checkpoint(pTest.model.prefix, pTest.model.epoch)
    pModel.process_weight(sym, arg_params, aux_params)
    b = pGen.batch_image
    s, l = shape
    net, params = relay.frontend.from_mxnet(
        sym, 
        dict(data=(b, 3, l, s), im_info=(b, 3), rec_id=(b, ), im_id=(b, )), 
        arg_params=arg_params, 
        aux_params=aux_params)

    num_threads = 56
    os.environ["TVM_NUM_THREADS"] = str(num_threads)
    target = "llvm -mcpu=core-avx2"

    with autotvm.apply_history_best(pTest.model.prefix.replace("checkpoint", "tune_x86.log")):
        with relay.build_config(opt_level=3):
            graph, lib, params = relay.build(net[net.entry_func], target, params=params)

    exe = graph_runtime.create(graph, lib, ctx)
    exe.set_input(**params)
    exe.set_input(**inputs)
     
    # run once
    exe.run()
    exe.get_output(0)

    # # evaluate
    # ftimer = exe.module.time_evaluator("run", ctx, number=1, repeat=count)
    # prof_res = np.array(ftimer().results) * 1000  # convert to millisecond
    # print("Mean inference time (std dev): %.2f ms (%.2f ms)" %
    #         (np.mean(prof_res), np.std(prof_res)))
