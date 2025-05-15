echo "hpsv2"
# python evaluation.py --data-type test --data-path /data/yangyuanming/dataset_temp/HPDv2 --image-path /data/yangyuanming/dataset_temp/HPDv2/test --checkpoint /mnt/jfs/yym/ckpt/HPSv2/HPS_v2.pt

echo "ImageReward"
python evaluation.py --data-type test --data-path /data/yangyuanming/dataset_temp/HPDv2 --image-path /data/yangyuanming/dataset_temp/HPDv2/test --checkpoint /mnt/jfs/yym/ckpt/ImageReward/ImageReward.pt