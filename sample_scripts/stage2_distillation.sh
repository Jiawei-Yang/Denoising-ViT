# Stage 2: distill denoised samples into a generalizable denoiser
# choose one of the following:
# vit_types=(
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
vit_type=vit_base_patch14_dinov2.lvd142m

DATE=`date "+%m%d"`
num_iters=100000 # 10 epochs * 10,000 samples = 100,000 iterations
if [[ $vit_type == *"patch14"* ]]; then
    input_size=518
    stride=14
else
    input_size=512
    stride=16
fi
python -m torch.distributed.launch --nproc_per_node=8 train_denoiser.py \
    --project ${DATE}_voc_distillation \
    --run_name clip \
    --config configs/distillation.yaml \
    vit_type=$vit_type \
    stride=$stride \
    data.data_list=data/voc_train_${vit_type}_s${stride}.txt \
    data.input_size=[$input_size,$input_size] \
    optim.num_iters=$num_iters \
    logging.vis_freq=2000 \
    logging.save_freq=2000 \
    optim.batch_size=8 # the effective batch size is 8*8=64