#!/usr/bin/env bash
# 无限循环版本，修改为每次增加50，遇到错误等待10分钟后重新执行

ckpt=350
max_ckpt=1000

while [ $ckpt -le $max_ckpt ]; do
    echo "[$(date '+%F %T')] 启动 python new_eval_pose.py ..."
    python new_eval_pose.py --checkpoint /home/mar/ckpt/DIC_S_25852/checkpoints/${ckpt}.ckpt
    if [ $? -eq 0 ]; then
        echo "[$(date '+%F %T')] 训练完成，继续下一个ckpt."
        ((ckpt+=50))  # 增加50
        sleep 15
    else
        echo "[$(date '+%F %T')] 发生错误，等待10分钟后重试ckpt=${ckpt} ..."
        sleep 600  # 等待10分钟
    fi
done
