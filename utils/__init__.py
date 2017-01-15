"""
This folder is for all scripts which do not import Theano
Basic small functions which can be reused
"""

import gzip
import platform
import random
import time

import numpy as np
from .paths import *

MAX_FLOAT = np.finfo(np.float32).max
MIN_FLOAT = np.finfo(np.float32).min
MAX_INT = np.iinfo(np.int32).max
MIN_INT = np.iinfo(np.int32).min

def hms(seconds):
    seconds = np.floor(seconds)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)

    return "%02d:%02d:%02d" % (hours, minutes, seconds)

def timestamp():
    return time.strftime("%Y%m%d-%H%M%S", time.localtime())


def hostname():
    return platform.node()


def generate_expid(arch_name):
    # expid shouldn't matter on anything else than configuration name.
    # Configurations need to be deterministic!
    return "%s" % (arch_name, )


def get_git_revision_hash():
    try:
        import subprocess
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], stderr=open('/dev/null')).strip()
    except:
        return "-1"

def current_learning_rate(schedule, idx):
    s = schedule.keys()
    s.sort()
    current_lr = schedule[0]
    for i in s:
        if idx >= i:
            current_lr = schedule[i]
    return current_lr


def detect_nans(loss, xs_shared, ys_shared, all_params):
    found = False
    if np.isnan(loss).any():
        print "NaN Detected."
        print "loss:", loss
        found=True
        #if not np.isfinite(g_n): print "Nan in gradients detected"
    for p in all_params:
        if not np.isfinite(p.get_value()).all():
            print "Nan detected in", p.name
            found=True
    for k, v in xs_shared.iteritems():
        if not np.isfinite(v.get_value()).all():
            print "Nan detected in loaded data: %s"%k
            found=True
    for k, v in ys_shared.iteritems():
        if not np.isfinite(v.get_value()).all():
            print "Nan detected in loaded data: %s"%k
            found=True
    if found:
        raise Exception("NaN's")

def varname(obj, namespace):
    r = [name for name in namespace if namespace[name] is obj]
    if len(r)==1:
        return r[0]
    else:
        return str(r)

def put_in_the_middle(target_tensor, data_tensor, pad_with_edge=False, padding_mask=None):
    """
    put data_sensor with arbitrary number of dimensions in the middle of target tensor.
    if data_tensor is bigger, data is cut off
    if target_sensor is bigger, original values (probably zeros) are kept
    :param target_tensor:
    :param data_tensor:
    :return:
    """
    target_shape = target_tensor.shape
    data_shape = data_tensor.shape

    def get_indices(target_width, data_width):
        if target_width>data_width:
            diff = target_width - data_width
            target_slice = slice(diff/2, target_width-(diff-diff/2))
            data_slice = slice(None, None)
        else:
            diff = data_width - target_width
            data_slice = slice(diff/2, data_width-(diff-diff/2))
            target_slice = slice(None, None)
        return target_slice, data_slice

    t_sh = [get_indices(l1,l2) for l1, l2 in zip(target_shape, data_shape)]
    target_indices, data_indices = zip(*t_sh)

    # do the copy
    target_tensor[target_indices] = data_tensor[data_indices]

    """
    check this code

    if padding_mask is not None:
        padding_mask[:] = True
        padding_mask[target_indices] = False

    if pad_with_edge:
        if target_indices[0].start:
            for i in xrange(0, target_indices[0].start):
                target_tensor[i] = data_tensor[0]
        if target_indices[0].stop:
            for i in xrange(target_indices[0].stop, len(target_tensor)):
                target_tensor[i] = data_tensor[-1]
    """
    return target_tensor

def put_randomly(target_tensor, data_tensor, seed=None):

    """
    put data_sensor with arbitrary number of dimensions in the middle of target tensor.
    if data_tensor is bigger, data is cut off
    if target_sensor is bigger, original values (probably zeros) are kept
    :param target_tensor:
    :param data_tensor:
    :return:
    """

    target_shape = target_tensor.shape
    data_shape = data_tensor.shape
    if seed:
        randomstate = random.getstate()
        random.seed(seed)

    def get_indices(target_width, data_width):
        if target_width>data_width:
            diff = target_width - data_width
            delta = random.randint(0,diff)
            target_slice = slice(delta, target_width-(diff-delta))
            data_slice = slice(None, None)
        elif target_width<data_width:
            diff = data_width - target_width
            delta = random.randint(0,diff)
            data_slice = slice(delta, data_width-(diff-delta))
            target_slice = slice(None, None)
        else:
            target_slice = slice(None, None)
            data_slice = slice(None, None)
        return target_slice, data_slice

    t_sh = [get_indices(l1,l2) for l1, l2 in zip(target_shape, data_shape)]
    target_indices, data_indices = zip(*t_sh)

    # do the copy
    target_tensor[target_indices] = data_tensor[data_indices]

    if seed:
        random.setstate(randomstate)
