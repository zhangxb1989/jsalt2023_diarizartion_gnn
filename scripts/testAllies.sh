#!/bin/bash

data_dir="data/ALLIES/embeddings/pkl/"
seg_dir="data/ALLIES/segment/"
rttm_dir="data/ALLIES/rttm_pred/"

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
        --model_filename checkpoint/voxceleb_sampler_epoch500_whole.pth \
        --knn_k 5 \
        --tau 0.05 \
        --level 10 \
        --threshold prob \
        --faiss_gpu \
        --hidden 256 \
        --num_conv 1 \
        --gat \
        --batch_size 4096 \
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

# 获取结束时间
end_time=$(date +%s)

# 计算总执行时间
total_duration=$((end_time - start_time))

# 打印总执行时间
echo "所有脚本执行完成，总耗时：$total_duration 秒"
