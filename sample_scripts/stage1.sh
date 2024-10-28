
conda activate dvt
start_idx=$1
num_imgs_per_gpu=$2
model_name=vit_base_patch14_dinov2.lvd142m
img_path="data/voc_train.txt"

for i in $(seq 0 7); do
    CUDA_VISIBLE_DEVICES=$i python main_img_denoising.py \
    --num_views 768 \
    --img_path $img_path \
    --data_root data/VOCdevkit \
    --save_root data/denoised_feats/voc/ \
    --output_dir work_dirs/stage-1/voc/${model_name} \
    --start_idx $((start_idx + i * num_imgs_per_gpu)) \
    --num_imgs $num_imgs_per_gpu \
    --input_size 518 518 \
    --model ${model_name} &
done
wait;