#!/usr/bin/env bash



function runexp {

dataset=${1}
batch_size=${2}




#--resume ${decompose_type}/cr${cr}/model_best.pth.tar

expname=/nfshomes/xliu1231/Causal_Disentangle/outputs/example


python -m torch.distributed.launch --nproc_per_node=4 celebA_exp.py  --workers 4  --epoch-num 20  --print_freq 10 --learning-rate 0.05 #> ${expname}.log 2>&1

}

#         dataset   batch_size
runexp    "CelebA"    128 
