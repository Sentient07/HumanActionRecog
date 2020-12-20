#embedding_utils.py



import numpy as np
import tensorboard_logger
import os.path as osp
import os

class Tb_logger(object):

    def __init__(self):
        pass

    def init_logger(self, path, splits):

        tb_logger = {}
        for split in splits:
            tb_logger[split] = tensorboard_logger.Logger(osp.join(path, split), flush_secs=5, dummy_time=1)

        return tb_logger


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=0):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / (.0001 + self.count)

    def __str__(self):
        """String representation for logging
        """
        # for values that should be recorded exactly e.g. iteration number
        if self.count == 0:
            return str(self.val)
        # for stats
        return '%.4f (%.4f)' % (self.val, self.avg)
