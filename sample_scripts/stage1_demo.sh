# Stage 1: denoise a single image
conda activate dvt
# choose one of the following:
vit_types=(
    # DINOv2
    "vit_base_patch14_dinov2.lvd142m"
    "vit_large_patch14_dinov2.lvd142m"
    # DINOv2 + register
    "vit_base_patch14_reg4_dinov2.lvd142m"
    # DEiT-III
    "deit3_base_patch16_224.fb_in1k"
    # CLIP
    "vit_base_patch16_clip_384.laion2b_ft_in12k_in1k"
    # EVA
    "eva02_base_patch16_clip_224.merged2b"
    # Auto-auged supervised ViT:
    "vit_base_patch16_384.augreg_in21k_ft_in1k"
    # DINO and MAE sometimes have artifacts in huge resolution
    # e.g., 720p, and with a stride of 8/16
    # DINO
    "vit_base_patch16_224.dino"
    # MAE
    "vit_base_patch16_224.mae"
)
num_iters=10000 # we use 20000 iterations in the paper
for vit_type in "${vit_types[@]}"; do
    expname=$vit_type
    # if you have > 30GB GPU memory, you can reduce the stride size to 7 or 8
    if [[ $vit_type == *"patch14"* ]]; then
        input_size=518
        stride_size=14
    else
        input_size=512
        stride_size=16
    fi
    python main_img_denoising.py \
        --num_views 768 \
        --img_path "demo/cat.jpg" \
        --output_dir work_dirs/demo/${model_name} \
        --input_size 518 518 \
        --model ${vit_type}
done
