#!/bin/bash

data_dir="data/ALLIES/embeddings/pkl/"
seg_dir="data/ALLIES/segment/"
rttm_dir="data/ALLIES/rttm_pred/"
rttm_gt_dir="data/ALLIES/rttm_gt/"

# 获取开始时间
start_time=$(date +%s)

# 遍历 data_dir 目录下的所有 pkl 文件
for data_file_path in $data_dir*.pkl; do
    # 提取文件名
    data_file_name=$(basename "$data_file_path")

    # 提取对应的 seg 文件名
    seg_file_name="${data_file_name%.pkl}.seg"

    # 提取对应的 rttm_pred 文件名
    rttm_file_name="${data_file_name%.pkl}.rttm"

    # 构建 seg 文件的完整路径
    seg_file_path="${seg_dir}${seg_file_name}"

    # 构建 rttm_pred 文件的完整路径
    rttm_file_path="${rttm_dir}${rttm_file_name}"

    # 获取当前循环开始时间
    loop_start_time=$(date +%s)

    # 构建命令行
    command="python test_subg.py \
        --data_path $data_file_path \
        --seg_path $seg_file_path \
        --rttm_pred_path $rttm_file_path \
        --model_filename checkpoint/voxceleb_split_200_level2.pth \
        --knn_k 8 \
        --tau 0.6 \
        --level 2 \
        --threshold prob \
        --faiss_gpu \
        --hidden 256 \
        --num_conv 1 \
        --use_cluster_feat \
        --batch_size 1024 \
        --early_stop"

    # 执行命令行
    $command

    # 获取当前循环结束时间
    loop_end_time=$(date +%s)

    # 计算当前循环执行时间
    loop_duration=$((loop_end_time - loop_start_time))

    # 打印当前循环执行时间
    echo "执行脚本 $data_file_name 完成，耗时：$loop_duration 秒"
done

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

# 获取结束时间
end_time=$(date +%s)

# 计算总执行时间
total_duration=$((end_time - start_time))

# 打印总执行时间
echo "所有脚本执行完成，总耗时：$total_duration 秒"
