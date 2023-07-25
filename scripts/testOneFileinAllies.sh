#python test_subg.py \
#--data_path data/ALLIES/embeddings/pkl/19981207.0700.inter_fm_dga.pkl \
#--seg_path data/ALLIES/segment/19981207.0700.inter_fm_dga.seg \
#--rttm_pred_path data/ALLIES/rttm_pred/19981207.0700.inter_fm_dga.rttm \
#--model_filename checkpoint/voxceleb_split_200_level2.pth \
#--knn_k 8 \
#--tau 0.5 \
#--level 2 \
#--threshold prob \
#--faiss_gpu \
#--hidden 256 \
#--num_conv 1 \
#--batch_size 1024 \
#--use_cluster_feat \
#--early_stop

if [ -f data/ALLIES/rttm_gt/19981207.0700.inter_fm_dga.rttm ]
then
    # run dscore
    python dscore/score.py -r data/ALLIES/rttm_gt/19981207.0700.inter_fm_dga.rttm -s data/ALLIES/rttm_pred/19981207.0700.inter_fm_dga.rttm --collar 0.25 --ignore_overlaps
fi
#--knn_k 5 \
#--tau 0.08 \
#--level 10 \
#--threshold prob \
#--faiss_gpu \
#--hidden 256 \
#--num_conv 1 \
#--gat \
#--batch_size 4096 \
#--early_stop