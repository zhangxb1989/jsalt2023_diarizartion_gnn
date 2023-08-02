#!/bin/bash

data_dir="data/ALLIES/embeddings/pkl/"
seg_dir="data/ALLIES/segment/"
rttm_dir="data/ALLIES/rttm_pred/"
rttm_gt_dir="data/ALLIES/rttm_gt/"
# 获取 rttm_dir 目录下的所有文件名
rttm_pred_files=$(ls -1 $rttm_dir)
# 构建 -s 参数
s_param=""
for rttm_pred_file in $rttm_pred_files; do
    s_param+=" $rttm_dir$rttm_pred_file"
    echo "$rttm_pred_file"
done

# 构建 -r 参数，基于 -s 参数的文件名
r_param=""
for rttm_pred_file in $rttm_pred_files; do
    r_file_name="${rttm_pred_file%.*}.rttm"
    r_param+=" $rttm_gt_dir$r_file_name"
done

# 执行 dscore
# shellcheck disable=SC2086
python dscore/score.py -r $r_param -s $s_param --collar 0.00 --ignore_overlaps