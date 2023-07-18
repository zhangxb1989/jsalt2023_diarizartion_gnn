python test_subg.py \
--data_path data/ALLIES/embeddings/19981207.0700.inter_fm_dga.pkl \
--model_filename checkpoint/vox_cp20_epoch250.pth \
--knn_k 10 \
--tau 0.08 \
--level 10 \
--threshold prob \
--faiss_gpu \
--hidden 256 \
--num_conv 1 \
--gat \
--batch_size 1024 \
--early_stop

if [ -f data/ALLIES/rttm_gt/19981207.0700.inter_fm_dga.rttm ]
then
    # run dscore
    python dscore/score.py -r data/ALLIES/rttm_gt/19981207.0700.inter_fm_dga.rttm -s data/ALLIES/rttm_pred/19981207.0700.inter_fm_dga.rttm --collar 0.25 --ignore_overlaps
fi
