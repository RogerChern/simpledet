import datetime
import time
import logging
import numpy as np
import mxnet as mx


class Speedometer(object):
    def __init__(self, batch_size, total_iter, frequent=50):
        self.batch_size = batch_size
        self.total_iter = total_iter
        self.frequent = frequent
        self.init = False
        self.tic = 0
        self.last_count = 0

    def __call__(self, param):
        """Callback to Show speed."""
        count = param.nbatch
        if self.last_count > count:
            self.init = False
        self.last_count = count

        if self.init:
            if count % self.frequent == 0:
                speed = self.frequent * self.batch_size / (time.time() - self.tic)
                eta = int((time.time() - self.tic) * (self.total_iter - param.iter) / self.frequent)
                if param.eval_metric is not None:
                    name, value = param.eval_metric.get()
                    s = "Epoch[%d] Batch [%d]\tIter: %d/%d\tLr: %.5f\tSpeed: %.2f samples/s\tETA: %s(%ds)\t" % \
                        (param.epoch, count, param.iter, self.total_iter, param.lr, speed, datetime.timedelta(seconds=eta), eta)
                    for n, v in zip(name, value):
                        s += "%s=%f,\t" % (n, v)
                    logging.info(s)
                else:
                    logging.info("Epoch[%d] Batch [%d]\tSpeed: %.2f samples/sec",
                                 param.epoch, count, speed)
                self.tic = time.time()
        else:
            self.init = True
            self.tic = time.time()

class DetailSpeedometer(object):
    def __init__(self, batch_size, frequent=50):
        self.batch_size = batch_size
        self.frequent = frequent
        self.init = False
        self.tic = 0
        self.last_count = 0

    def __call__(self, param):
        """Callback to Show speed."""
        count = param.nbatch
        rank = param.rank
        total_iter = param.total_iter

        if self.last_count > count:
            self.init = False
        self.last_count = count

        if self.init:
            if count % self.frequent == 0:
                speed = self.frequent * self.batch_size / (time.time() - self.tic)
                if param.eval_metric is not None:
                    name, value = param.eval_metric.get()
                    s = "Epoch[%d] Rank[%d] Batch[%d] TotalIter[%d] Train:%.3f(%.3f)\tkv_sync:%.3f(%.3f)\t" \
                        "data:%.3f(%.3f)\titer_total_time:%.3f(%.3f)\tSpeed: %.2f samples/sec\tTrain-" % (
                        param.epoch, rank, count, total_iter,
                        param.cur_batch_time, param.avg_batch_time,
                        param.cur_kvstore_sync_time, param.avg_kvstore_sync_time,
                        param.cur_data_time, param.avg_data_time,
                        param.cur_iter_total_time, param.avg_iter_total_time,
                        speed)
                    for n, v in zip(name, value):
                        s += "%s=%f,\t" % (n, v)
                    logging.info(s)
                else:
                    logging.info(
                        "Epoch[%d] Rank[%d] Batch[%d] TotalIter[%d] Train:%.3f(%.3f)\tkv_sync:%.3f(%.3f)\tdata:%.3f(%.3f)\titer_total_time:%.3f(%.3f)\tSpeed: %.2f samples/sec",
                        param.epoch, rank, count, total_iter,
                        param.cur_batch_time, param.avg_batch_time,
                        param.cur_kvstore_sync_time, param.avg_kvstore_sync_time,
                        param.cur_data_time, param.avg_data_time,
                        param.cur_iter_total_time, param.avg_iter_total_time,
                        speed)

                self.tic = time.time()
        else:
            self.init = True
            self.tic = time.time()


def do_checkpoint(prefix):
    def _callback(iter_no, sym, arg, aux, arg1=None, aux1=None):
        mx.model.save_checkpoint(prefix, iter_no + 1, sym, arg, aux)
        if arg1 is not None and aux1 is not None:
            mx.model.save_checkpoint(prefix + "_ema", iter_no + 1, sym, arg1, aux1)
    return _callback

