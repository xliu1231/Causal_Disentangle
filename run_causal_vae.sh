#!/usr/bin/env bash



function runexp {

dataset=${1}
batch_size=${2}
z_dim=${3}




#--resume ${decompose_type}/cr${cr}/model_best.pth.tar

expname=/nfshomes/xliu1231/Causal_Disentangle/outputs/causalvae_${z_dim}


python -m torch.distributed.launch --nproc_per_node=4 celebA_causalvae_exp.py  --workers 4  --epoch-num 30  --print_freq 30 --learning-rate 0.05 --batch-size=${batch_size} --z-dim=${z_dim} #> ${expname}.log 2>&1

}

#         dataset   batch_size  z_dim
runexp    "CelebA"    128    128
