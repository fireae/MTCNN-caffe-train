#!/usr/bin/env sh
set -e
CAFFE_ROOT=
${CAFFE_ROOT}/build/tools/caffe train -solver ./ONet_solver.prototxt


