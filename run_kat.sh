#!/usr/bin/env bash


THEANO_FLAGS='device=gpu0,floatX=float32,allow_gc=True' python test_seg_scan_dsb_prl.py dsb_s5_p8a1 0
THEANO_FLAGS='device=gpu1,floatX=float32,allow_gc=True' python test_seg_scan_dsb_prl.py dsb_s5_p8a1 1
THEANO_FLAGS='device=gpu2,floatX=float32,allow_gc=True' python test_seg_scan_dsb_prl.py dsb_s5_p8a1 2
THEANO_FLAGS='device=gpu3,floatX=float32,allow_gc=True' python test_seg_scan_dsb_prl.py dsb_s5_p8a1 3

THEANO_FLAGS='device=gpu2,floatX=float32,allow_gc=True' python test_fpred_scan_dsb.py dsb_c3_s5_p8a1

