for i in {1..20}
do
    data_path="data/VoxCeleb/split/split20/voceleb_split${i}.pkl"
    model_filename="checkpoint/vox_cp${i}.pth"
    temp=$((i-1))
    p_model_filename="checkpoint/vox_cp${temp}.pth"

    if [ ! -f "$p_model_filename" ] && [ "$i" -ne 1 ]; then
        echo "File does not exist. Breaking out of the loop."
        break
    fi
    python train_subg_loop.py --data_path $data_path --p_model_filename $p_model_filename --model_filename $model_filename --knn_k 10 --levels 2 --faiss_gpu --hidden 256 --epochs 200 --lr 0.01 --batch_size 2048 --num_conv 1 --balance --use_cluster_feat
    echo "sleeping 5 mins"
    sleep 300  # 300 秒等于 5 分钟
done