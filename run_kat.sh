#!/usr/bin/env bash
set -x
sleep 6h
THEANO_FLAGS='device=gpu2,floatX=float32,allow_gc=True' python train_fpred_patch.py luna_c4
