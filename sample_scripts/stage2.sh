

conda activate dvt

project="stage-2"
run_name="vit_base_patch14_dinov2.lvd142m"
model="vit_base_patch14_dinov2.lvd142m"

torchrun --nproc_per_node=8 \
    main_denoiser.py \
    --model $model \
    --data_root data/VOCdevkit \
    --feat_root data/denoised_feats/voc/denoised_features/$model \
    --data_list_path data/voc_train.txt \
    --auto_stride \
    --num_iterations 50000 \
    --project $project \
    --run_name ${run_name} \
    --batch_size 16 \
    --num_workers 16


# evaluation

conda activate dvt_eval
ckpt_pth="work_dirs/${project}/${run_name}/checkpoints/ckpt_049999.pth"

# VOC
echo "start evalution on VOC"
workdir="./work_dirs_eval/${project}/${run_name}/voc_seg/"
config="evaluation/configs/vitb_voc2012_linear_config.py"
torchrun --nproc_per_node=8 \
    evaluate_dense_tasks.py \
    ${config} \
    --backbone-type $model \
    --task segmentation \
    --work-dir ${workdir} \
    --load-denoiser-from ${ckpt_pth} 


# ADE20k
echo "start evalution on ADE20k"
config="evaluation/configs/vitb_ade20k_linear_config.py"
workdir="./work_dirs_eval/${project}/${run_name}/ade/"
torchrun --nproc_per_node=8  \
    evaluate_dense_tasks.py  \
    ${config} \
    --backbone-type $model \
    --task segmentation \
    --work-dir ${workdir} \
    --load-denoiser-from ${ckpt_pth} 

# NYUv2
echo "start evalution on NYUv2"
config="evaluation/configs/vitb_nyu_linear_config.py"
workdir="./work_dirs_eval/${project}/${run_name}/nyu/"
torchrun --nproc_per_node=8  \
    evaluate_dense_tasks.py  \
    ${config} \
    --backbone-type $model \
    --task depth \
    --work-dir ${workdir} \
    --load-denoiser-from ${ckpt_pth} 