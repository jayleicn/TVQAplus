#!/usr/bin/env bash
model_dir=$1
mode=$2
python inference.py \
--model_dir=${model_dir} \
--mode=${mode} \
${@:3}
