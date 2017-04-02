#!/bin/bash

21.03
THEANO_FLAGS='device=gpu2,floatX=float32,allow_gc=True' python test_seg_scan_dsb_prl.py  dsb_s5_p8a1 3
THEANO_FLAGS='device=gpu3,floatX=float32,allow_gc=True' python train_class_dsb.py dsb_a05_c3_s1e_p8a1

