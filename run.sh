#!/bin/bash

THEANO_FLAGS='device=gpu1,floatX=float32' python test_seg_scan_dsb_prl.py  dsb_s5_p8a1 0 # finished!!!

THEANO_FLAGS='device=gpu0,floatX=float32,allow_gc=True' python test_seg_scan_dsb_prl.py  dsb_s5_p8a1 2 # finished
THEANO_FLAGS='device=gpu1,floatX=float32,allow_gc=True' python test_seg_scan_dsb_prl.py  dsb_s5_p8a1 1 #finished
THEANO_FLAGS='device=gpu2,floatX=float32,allow_gc=True' python test_seg_scan_dsb_prl.py  dsb_s5_p8a1 3
