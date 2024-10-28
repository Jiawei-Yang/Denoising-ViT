#!/bin/bash

# choose model from the following:
# models=(
#     # DINOv2
#     "vit_base_patch14_dinov2.lvd142m"
#     # DINOv2 + register
#     "vit_base_patch14_reg4_dinov2.lvd142m"
#     # DEiT-III
#     "deit3_base_patch16_224.fb_in1k"
#     # CLIP
#     "vit_base_patch16_clip_384.laion2b_ft_in12k_in1k"
#     # EVA
#     "eva02_base_patch16_clip_224.merged2b"
#     # DINO
#     "vit_base_patch16_224.dino"
#     # MAE
#     "vit_base_patch16_224.mae"
# )
denoiser_ckpt="Your checkpoint path"
model="vit_base_patch14_dinov2.lvd142m"
ngpu=8
if [[ $vit_type == *"patch14"* ]]; then
    stride_size=14
else
    stride_size=16
fi

# VOC2012:
config="evaluation/configs/vitb_voc2012_linear_config.py"
workdir="./work_dirs/segmentation_eval/voc2012/voc10k_distilled/${model}"
bash evaluation/scripts/dist_eval_segmentation.sh \
    "evaluation/configs/vitb_voc2012_linear_config.py" ${ngpu} \
    --backbone-type $model \
    --stride ${stride_size} \
    --work-dir ${workdir} \
    --load-denoiser-from $denoiser_ckpt

# ADE20k
config="evaluation/configs/vitb_ade20k_linear_config.py"
workdir="./work_dirs/segmentation_eval/ade20k/voc10k_distilled/${model}"
bash evaluation/scripts/dist_eval_segmentation.sh \
    ${config} ${ngpu} ${ngpu} \
    --backbone-type $model \
    --stride ${stride_size} \
    --work-dir ${workdir} \
    --load-denoiser-from $denoiser_ckpt

# depth
config="evaluation/configs/vitb_nyu_linear_norm5e-3_config.py"
workdir="./work_dirs/depth_eval/nyu_paper/voc10k_distilled/${model}"
bash evaluation/scripts/dist_eval_depth.sh \
    ${config} ${ngpu} \
    --backbone-type $model \
    --stride ${stride_size} \
    --work-dir ${workdir} \
    --load-denoiser-from $denoiser_ckpt