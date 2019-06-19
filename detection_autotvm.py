import argparse
import importlib
import os
import time

from utils.load_model import load_checkpoint

import numpy as np
import mxnet as mx
import tvm
import tvm.relay.testing
import tvm.contrib.graph_runtime as runtime
from tvm import autotvm, relay
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner
from tvm.contrib.util import tempdir


def parse_args():
    parser = argparse.ArgumentParser(description='Autotune detectors for inference')
    # general
    parser.add_argument('--config', help='config file path', type=str, required=True)
    parser.add_argument('--shape', help='specify input 2d image shape', metavar=('SHORT', 'LONG'), type=int, nargs=2, required=True)
    parser.add_argument('--device', help='target device, one of x86 or cuda', type=str, required=True)
    parser.add_argument('--gpu', help='GPU index', type=int, default=0)
    args = parser.parse_args()

    config = importlib.import_module(args.config.replace('.py', '').replace('/', '.'))
    return config, args.device, args.gpu, args.shape


def tune_cuda(tasks,
              measure_option,
              tuner='xgb',
              n_trial=1000,
              early_stopping=None,
              log_filename='tuning.log',
              use_transfer_learning=True,
              try_winograd=True):
    if try_winograd:
        for i in range(len(tasks)):
            try:  # try winograd template
                tsk = autotvm.task.create(tasks[i].name, tasks[i].args,
                                          tasks[i].target, tasks[i].target_host, 'winograd')
                input_channel = tsk.workload[1][1]
                if input_channel >= 64:
                    tasks[i] = tsk
            except Exception:
                pass

    # create tmp log file
    tmp_log_file = log_filename + ".tmp"
    if os.path.exists(tmp_log_file):
        os.remove(tmp_log_file)

    for i, tsk in enumerate(reversed(tasks)):
        prefix = "[Task %2d/%2d] " %(i+1, len(tasks))
        print(tsk)

        # task.args -> input -> shape -> batch size
        # large batch size indicates convs in detection head
        # tune longer for this convs
        if tsk.args[0][1][0] >= 50:
            task_n_trail = n_trial * 3
            task_early_stop = None
        else:
            task_n_trail = n_trial
            task_early_stop = early_stopping

        # create tuner
        if tuner == 'xgb' or tuner == 'xgb-rank':
            tuner_obj = XGBTuner(tsk, loss_type='rank')
        elif tuner == 'ga':
            tuner_obj = GATuner(tsk, pop_size=100)
        elif tuner == 'random':
            tuner_obj = RandomTuner(tsk)
        elif tuner == 'gridsearch':
            tuner_obj = GridSearchTuner(tsk)
        else:
            raise ValueError("Invalid tuner: " + tuner)

        if use_transfer_learning:
            if os.path.isfile(tmp_log_file):
                tuner_obj.load_history(autotvm.record.load_from_file(tmp_log_file))

        # do tuning
        tuner_obj.tune(n_trial=min(task_n_trail, len(tsk.config_space)),
                       early_stopping=task_early_stop,
                       measure_option=measure_option,
                       callbacks=[
                           autotvm.callback.progress_bar(task_n_trail, prefix=prefix),
                           autotvm.callback.log_to_file(tmp_log_file)])

    # pick best records to a cache file
    autotvm.record.pick_best(tmp_log_file, log_filename)
    os.remove(tmp_log_file)


def tune_x86(tasks,
             measure_option,
             tuner='gridsearch',
             early_stopping=None,
             log_filename='tuning.log'):

    for i, tsk in enumerate(tasks):
        prefix = "[Task %2d/%2d] " % (i+1, len(tasks))

        # converting conv2d tasks to conv2d_NCHWc tasks
        op_name = tsk.workload[0]
        if op_name == 'conv2d':
            func_create = 'topi_x86_conv2d_NCHWc'
        elif op_name == 'depthwise_conv2d_nchw':
            func_create = 'topi_x86_depthwise_conv2d_NCHWc_from_nchw'
        else:
            raise ValueError("Tuning {} is not supported on x86".format(op_name))

        task = autotvm.task.create(func_create, args=tsk.args,
                                   target=target, template_key='direct')
        task.workload = tsk.workload

        # create tuner
        if tuner == 'xgb' or tuner == 'xgb-rank':
            tuner_obj = XGBTuner(task, loss_type='rank')
        elif tuner == 'ga':
            tuner_obj = GATuner(task, pop_size=50)
        elif tuner == 'random':
            tuner_obj = RandomTuner(task)
        elif tuner == 'gridsearch':
            tuner_obj = GridSearchTuner(task)
        else:
            raise ValueError("Invalid tuner: " + tuner)

        # do tuning
        n_trial=len(task.config_space)
        tuner_obj.tune(n_trial=n_trial,
                       early_stopping=early_stopping,
                       measure_option=measure_option,
                       callbacks=[
                           autotvm.callback.progress_bar(n_trial, prefix=prefix),
                           autotvm.callback.log_to_file(log_filename)])


if __name__ == "__main__":
    config, device, gpu, shape = parse_args()

    pGen, pKv, pRpn, pRoi, pBbox, pDataset, pModel, pOpt, pTest, \
        transform, data_name, label_name, metric_list = config.get_config(is_train=False)

    # create dummy data batch
    data = mx.nd.ones(shape=[1, 3] + shape)
    im_info = mx.nd.array([x / 2.0 for x in shape] + [2.0]).reshape(1, 3)
    im_id = mx.nd.array([1])
    rec_id = mx.nd.array([1])
    data_names = ["data", "im_info", "im_id", "rec_id"]
    ctx = tvm.gpu(gpu) if device == "cuda" else tvm.cpu()
    inputs = {k: tvm.nd.array(d.asnumpy(), ctx=ctx) for k, d in zip(data_names, [data, im_info, im_id, rec_id])}

    sym = pModel.test_symbol
    arg_params, aux_params = load_checkpoint(pTest.model.prefix, pTest.model.epoch)
    pModel.process_weight(sym, arg_params, aux_params)
    b = pGen.batch_image
    s, l = shape
    net, params = relay.frontend.from_mxnet(
        sym, 
        dict(data=(b, 3, l, s), im_info=(b, 3), rec_id=(b, ), im_id=(b, )), 
        arg_params=arg_params, 
        aux_params=aux_params)

    if device == "cuda":
        #### DEVICE CONFIG ####
        target = tvm.target.cuda()

        #### TUNING OPTION ####
        log_file = pTest.model.prefix.replace("checkpoint", "tune.log")
        dtype = 'float32'

        tuning_option = {
            'log_filename': log_file,

            'tuner': 'xgb',
            'n_trial': 2000,
            'early_stopping': 600,

            'measure_option': autotvm.measure_option(
                builder=autotvm.LocalBuilder(timeout=10),
                #runner=autotvm.LocalRunner(number=20, repeat=3, timeout=4, min_repeat_ms=150),
                runner=autotvm.RPCRunner(
                    '1080ti',  # change the device key to your key
                    '0.0.0.0', 9190,
                    number=20, repeat=3, timeout=20, min_repeat_ms=150)
            ),
        }

        print("Extract tasks...")
        tasks = autotvm.task.extract_from_program(net[net.entry_func], target=target,
                                            params=params, ops=(relay.op.nn.conv2d,))
        print("Tuning...")
        tune_cuda(tasks, **tuning_option)

    elif device == "x86":
        target = "llvm -mcpu=core-avx2"
        num_threads = 56
        os.environ["TVM_NUM_THREADS"] = str(num_threads)

        log_file = pTest.model.prefix.replace("checkpoint", "tune_x86.log")

        tuning_option = {
            'log_filename': log_file,
            'tuner': 'random',
            'early_stopping': None,

            'measure_option': autotvm.measure_option(
                builder=autotvm.LocalBuilder(),
                runner=autotvm.LocalRunner(number=10, repeat=1, min_repeat_ms=1000),
            ),
        }

        print("Extract tasks...")
        tasks = autotvm.task.extract_from_program(net[net.entry_func], target=target,
                                            params=params, ops=(relay.op.nn.conv2d,))
        print("Tuning...")
        tune_x86(tasks, **tuning_option)
