#!/usr/bin/env bash
hdfs dfs -get $PAI_DEFAULT_FS_URI/data/datasets/CUB_200_2011.tar
hdfs dfs -get $PAI_DEFAULT_FS_URI/data/models/zhangyu/resnet152-b121ed2d.pth
tar -xf ./CUB_200_2011.tar
#unzip -q ADEChallengeData2016.zip
python3 train.py
#python3 -m torch.distributed.launch --nproc_per_node=4 train2.py

