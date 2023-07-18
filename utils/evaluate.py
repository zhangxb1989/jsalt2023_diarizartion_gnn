#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import inspect
import os
import pandas as pd
import numpy as np
from clustering_benchmark import ClusteringBenchmark
from utils import metrics
from .misc import TextColors, Timer


# from utils import TextColors
# from utils import Timer


def _read_meta(fn):
    labels = list()
    lb_set = set()
    with open(fn) as f:
        for lb in f.readlines():
            lb = int(lb.strip())
            labels.append(lb)
            lb_set.add(lb)
    return np.array(labels), lb_set


def evaluate(gt_labels, pred_labels, metric="pairwise"):
    if isinstance(gt_labels, str) and isinstance(pred_labels, str):
        print("[gt_labels] {}".format(gt_labels))
        print("[pred_labels] {}".format(pred_labels))
        gt_labels, gt_lb_set = _read_meta(gt_labels)
        pred_labels, pred_lb_set = _read_meta(pred_labels)

        print(
            "#inst: gt({}) vs pred({})".format(len(gt_labels), len(pred_labels))
        )
        print(
            "#cls: gt({}) vs pred({})".format(len(gt_lb_set), len(pred_lb_set))
        )

    metric_func = metrics.__dict__[metric]

    with Timer(
            "evaluate with {}{}{}".format(TextColors.FATAL, metric, TextColors.ENDC)
    ):
        result = metric_func(gt_labels, pred_labels)
    if isinstance(result, np.float):
        print(
            "{}{}: {:.4f}{}".format(
                TextColors.OKGREEN, metric, result, TextColors.ENDC
            )
        )
    else:
        ave_pre, ave_rec, fscore = result
        print(
            "{}ave_pre: {:.4f}, ave_rec: {:.4f}, fscore: {:.4f}{}".format(
                TextColors.OKGREEN, ave_pre, ave_rec, fscore, TextColors.ENDC
            )
        )


def evaluation(pred_labels, labels, metrics):
    print("==> evaluation")
    # pred_labels = g.ndata['pred_labels'].cpu().numpy()
    max_cluster = np.max(pred_labels)
    # gt_labels_all = g.ndata['labels'].cpu().numpy()
    print("pre_cluster:", pred_labels)
    print("pre_shape:", len(pred_labels))
    gt_labels_all = labels
    pred_labels_all = pred_labels
    metric_list = metrics.split(",")
    for metric in metric_list:
        evaluate(gt_labels_all, pred_labels_all, metric)
    # H and C-scores
    gt_dict = {}
    pred_dict = {}
    for i in range(len(gt_labels_all)):
        gt_dict[str(i)] = gt_labels_all[i]
        pred_dict[str(i)] = pred_labels_all[i]
    bm = ClusteringBenchmark(gt_dict)
    scores = bm.evaluate_vmeasure(pred_dict)
    fmi_scores = bm.evaluate_fowlkes_mallows_score(pred_dict)
    print(scores)


# def write_output(file_name, fp, out_labels, starts, ends):
#     for label, seg_start, seg_end in zip(out_labels, starts, ends):
#         fp.write(f'SPEAKER {file_name} 1 {seg_start:03f} {seg_end - seg_start:03f} '
#                  f'<NA> <NA> {label + 1} <NA> <NA>{os.linesep}')
def write_pred_rttm(pred_labels):
    input_file = "data/ALLIES/segment/19981207.0700.inter_fm_dga.seg"
    output_file = "data/ALLIES/rttm_pred/19981207.0700.inter_fm_dga.rttm"
    # 读取文件
    df = pd.read_csv(input_file, sep='\s+', header=None)
    # 获取文件名（去除后缀）
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    # 构建新的DataFrame
    new_df = pd.DataFrame()
    new_df['Type'] = ['SPEAKER'] * len(df)
    new_df['Filename'] = [base_name] * len(df)
    new_df['SpeakerCount'] = ['1'] * len(df)
    new_df['ThirdColumn'] = round(df[2], 3)  # 我们假设输入文件的第三列对应的是df的第二列（因为pandas的列索引是从0开始的）
    new_df['Difference'] = round((df[3] - df[2]), 3)
    new_df['NA1'] = ['<NA>'] * len(df)
    new_df['NA2'] = ['<NA>'] * len(df)
    new_df['Random'] = pred_labels  # 生成随机数
    new_df['NA3'] = ['<NA>'] * len(df)
    new_df['NA4'] = ['<NA>'] * len(df)
    # 输出到新文件
    new_df.to_csv(output_file, sep=' ', index=False, header=None)
