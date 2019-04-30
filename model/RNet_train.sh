#!/usr/bin/env sh
set -e
CAFFE_ROOT=
${CAFFE_ROOT}/build/tools/caffe train -solver ./RNet_solver.prototxt -weights ./MTCNN/det2-24_iter_100000.caffemodel

