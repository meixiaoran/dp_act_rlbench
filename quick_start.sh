#!/usr/bin/env bash
# 无限循环版本
ckpt=100
max_ckpt=900

while [ $ckpt -le $max_ckpt ]; do
    echo "[$(date '+%F %T')] 启动 python new_eval_pose.py ..."
    python new_eval_pose.py --checkpoint /home/mar/ckpt/DIC_B/13.21.48_train_diffusion_transformer_hybrid_reach_target/checkpoints/${ckpt}.ckpt
    echo "[$(date '+%F %T')] train.py 已退出，等待 100 秒后重启 ..."
    sleep 15
    ((ckpt+=100))  # 增加100
done
