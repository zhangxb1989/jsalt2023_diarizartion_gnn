python test_subg.py \
--data_path data/ALLIES/embeddings/pkl/20111208.0650.BFMTV_PlaneteShowbiz.pkl \
--seg_path data/ALLIES/segment/20111208.0650.BFMTV_PlaneteShowbiz.seg \
--rttm_pred_path data/ALLIES/rttm_pred/20111208.0650.BFMTV_PlaneteShowbiz.rttm \
--model_filename checkpoint/voxceleb_split_200_c.pth \
--knn_k 10 \
--tau 0.5 \
--level 2 \
--threshold prob \
--faiss_gpu \
--hidden 256 \
--num_conv 1 \
--batch_size 1024 \
--use_cluster_feat \
--early_stop

if [ -f data/ALLIES/rttm_gt/20111208.0650.BFMTV_PlaneteShowbiz.rttm ]
then
    # run dscore
    python dscore/score.py -r data/ALLIES/rttm_gt/20111208.0650.BFMTV_PlaneteShowbiz.rttm -s data/ALLIES/rttm_pred/20111208.0650.BFMTV_PlaneteShowbiz.rttm --collar 0.00 --ignore_overlaps
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