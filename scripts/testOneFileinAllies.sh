python test_subg.py \
--data_path data/ALLIES/embeddings/pkl/20120510.0650.BFMTV_CultureEtVous.pkl \
--seg_path data/ALLIES/segment/20120510.0650.BFMTV_CultureEtVous.seg \
--rttm_pred_path data/ALLIES/rttm_pred/20120510.0650.BFMTV_CultureEtVous.rttm \
--model_filename checkpoint/voxceleb_speakers5994_150.pth \
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

if [ -f data/ALLIES/rttm_gt/20120510.0650.BFMTV_CultureEtVous.rttm ]
then
    # run dscore
    python dscore/score.py -r data/ALLIES/rttm_gt/20120510.0650.BFMTV_CultureEtVous.rttm -s data/ALLIES/rttm_pred/20120510.0650.BFMTV_CultureEtVous.rttm --collar 0.00 --ignore_overlaps
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