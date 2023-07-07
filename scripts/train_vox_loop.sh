for i in {1..20}
do
    data_path="data/VoxCeleb/split/voceleb_split${i}.pkl"
    model_filename="checkpoint/vox_cp${i}.pth"
    temp=$((i-1))
    p_model_filename="checkpoint/vox_cp${temp}.pth"

    if [ ! -f "$p_model_filename" ] && [ "$i" -ne 1 ]; then
        echo "File does not exist. Breaking out of the loop."
        break
    fi
    python train_subg_loop.py --data_path $data_path --p_model_filename $p_model_filename --model_filename $model_filename --knn_k 30 --levels 2,3 --faiss_gpu --hidden 256 --epochs 100 --lr 0.01 --batch_size 2048 --num_conv 1 --balance --gat
    echo "sleeping 5 mins"
    sleep 500  # 300 秒等于 5 分钟
done