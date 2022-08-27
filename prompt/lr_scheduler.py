"""
Modified from https://github.com/KaiyangZhou/deep-person-reid
"""
import paddle
from paddle.optimizer.lr import LRScheduler

AVAI_SCHEDS = ['single_step', 'multi_step', 'cosine']


class _BaseWarmupScheduler(LRScheduler):

    def __init__(
        self,
        learning_rate,
        successor,
        warmup_epoch,
        last_epoch=-1,
        verbose=False
    ):
        self.successor = successor
        self.warmup_epoch = warmup_epoch
        super().__init__(learning_rate,last_epoch, verbose)

    def get_lr(self):
        raise NotImplementedError

    def step(self, epoch=None):
        if self.last_epoch >= self.warmup_epoch:
            self.successor.step(epoch)
            self._last_lr = self.successor.get_last_lr()
        else:
            super().step(epoch)


class ConstantWarmupScheduler(_BaseWarmupScheduler):

    def __init__(
        self,
        learning_rate,
        successor,
        warmup_epoch,
        cons_lr,
        last_epoch=-1,
        verbose=False
    ):
        self.cons_lr = cons_lr
        super().__init__(
            learning_rate, successor, warmup_epoch, last_epoch, verbose
        )

    def get_lr(self):
        if self.last_epoch >= self.warmup_epoch:
            return self.successor.get_last_lr()
        return self.cons_lr


class LinearWarmupScheduler(_BaseWarmupScheduler):

    def __init__(
        self,
        learning_rate,
        successor,
        warmup_epoch,
        min_lr,
        last_epoch=-1,
        verbose=False
    ):
        self.min_lr = min_lr
        super().__init__(
            learning_rate, successor, warmup_epoch, last_epoch, verbose
        )

    def get_lr(self):
        if self.last_epoch >= self.warmup_epoch:
            return self.successor.get_last_lr()
        if self.last_epoch == 0:
            return self.min_lr
        return self.base_lr * self.last_epoch / self.warmup_epoch



def build_lr_scheduler(learning_rate, max_epoch, warmup_epoch, warmup_lr):
    """A function wrapper for building a learning rate scheduler.

    Args:
        learning_rate (Optimizer): learning_rate.
        optim_cfg (CfgNode): optimization config.
    """

    scheduler = paddle.optimizer.lr.CosineAnnealingDecay(
        learning_rate, float(max_epoch))

    scheduler.last_epoch = warmup_epoch

    if warmup_epoch > 0:
        scheduler = ConstantWarmupScheduler(
            learning_rate, scheduler, warmup_epoch, warmup_lr)


    return scheduler
