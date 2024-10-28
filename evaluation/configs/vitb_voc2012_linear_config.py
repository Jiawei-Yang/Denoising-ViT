custom_imports = dict(imports=["evaluation.segmentation"])
dataset_type = "PascalVOCDataset"
data_root = "data/VOCdevkit/VOC2012"
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (512, 512)
train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations"),
    dict(type="Resize", img_scale=(2048, 512), ratio_range=(0.5, 2.0)),
    dict(type="RandomCrop", crop_size=(512, 512), cat_max_ratio=0.75),
    dict(type="RandomFlip", prob=0.5),
    dict(type="PhotoMetricDistortion"),
    dict(
        type="Normalize",
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True,
    ),
    dict(type="Pad", size=(512, 512), pad_val=0, seg_pad_val=255),
    dict(type="DefaultFormatBundle"),
    dict(type="Collect", keys=["img", "gt_semantic_seg"]),
]
test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="MultiScaleFlipAug",
        img_scale=(2048, 512),
        flip=False,
        transforms=[
            dict(type="Resize", keep_ratio=True),
            dict(type="RandomFlip"),
            dict(
                type="Normalize",
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True,
            ),
            dict(type="ImageToTensor", keys=["img"]),
            dict(type="Collect", keys=["img"]),
        ],
    ),
]
data = dict(
    samples_per_gpu=2,  # 2,
    workers_per_gpu=6,
    train=dict(
        type="PascalVOCDataset",
        data_root=data_root,
        img_dir="JPEGImages",
        ann_dir="SegmentationClass",
        split="ImageSets/Segmentation/train.txt",
        pipeline=[
            dict(type="LoadImageFromFile"),
            dict(type="LoadAnnotations"),
            dict(type="Resize", img_scale=(2048, 512), ratio_range=(0.5, 2.0)),
            dict(type="RandomCrop", crop_size=(512, 512), cat_max_ratio=0.75),
            dict(type="RandomFlip", prob=0.5),
            dict(type="PhotoMetricDistortion"),
            dict(
                type="Normalize",
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True,
            ),
            dict(type="Pad", size=(512, 512), pad_val=0, seg_pad_val=255),
            dict(type="DefaultFormatBundle"),
            dict(type="Collect", keys=["img", "gt_semantic_seg"]),
        ],
    ),
    val=dict(
        type="PascalVOCDataset",
        data_root=data_root,
        img_dir="JPEGImages",
        ann_dir="SegmentationClass",
        split="ImageSets/Segmentation/val.txt",
        pipeline=[
            dict(type="LoadImageFromFile"),
            dict(
                type="MultiScaleFlipAug",
                img_scale=(2048, 512),
                flip=False,
                transforms=[
                    dict(type="Resize", keep_ratio=True),
                    dict(type="RandomFlip"),
                    dict(
                        type="Normalize",
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True,
                    ),
                    dict(type="ImageToTensor", keys=["img"]),
                    dict(type="Collect", keys=["img"]),
                ],
            ),
        ],
    ),
    test=dict(
        type="PascalVOCDataset",
        data_root=data_root,
        img_dir="JPEGImages",
        ann_dir="SegmentationClass",
        split="ImageSets/Segmentation/val.txt",
        pipeline=[
            dict(type="LoadImageFromFile"),
            dict(
                type="MultiScaleFlipAug",
                img_scale=(2048, 512),
                flip=False,
                transforms=[
                    dict(type="Resize", keep_ratio=True),
                    dict(type="RandomFlip"),
                    dict(
                        type="Normalize",
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True,
                    ),
                    dict(type="ImageToTensor", keys=["img"]),
                    dict(type="Collect", keys=["img"]),
                ],
            ),
        ],
    ),
)
log_config = dict(interval=50, hooks=[dict(type="TextLoggerHook", by_epoch=False)])
dist_params = dict(backend="nccl")
log_level = "INFO"
load_from = None
resume_from = None
workflow = [("train", 1)]
cudnn_benchmark = True
optimizer = dict(type="AdamW", lr=0.001, weight_decay=0.0001, betas=(0.9, 0.999))
optimizer_config = dict(
    type="DistOptimizerHook",
    update_interval=1,
    grad_clip=None,
    coalesce=True,
    bucket_size_mb=-1,
    use_fp16=False,
)
lr_config = dict(
    policy="poly",
    warmup="linear",
    warmup_iters=1500,
    warmup_ratio=1e-06,
    power=1.0,
    min_lr=0.0,
    by_epoch=False,
)
runner = dict(type="IterBasedRunner", max_iters=40000)
checkpoint_config = dict(by_epoch=False, interval=10000)
evaluation = dict(interval=10000, metric="mIoU", pre_eval=True)
fp16 = None
find_unused_parameters = False
norm_cfg = dict(type="SyncBN", requires_grad=True)
model = dict(
    type="EncoderDecoder",
    pretrained=None,
    backbone=dict(type="DinoVisionTransformer", out_indices=[8, 9, 10, 11]),
    decode_head=dict(
        type="BNHead",
        in_channels=[768],
        in_index=[3],
        input_transform="resize_concat",
        channels=768,
        dropout_ratio=0,
        num_classes=21,
        norm_cfg=dict(type="SyncBN", requires_grad=True),
        align_corners=False,
        loss_decode=dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0),
    ),
    test_cfg=dict(mode="slide", crop_size=(512, 512), stride=(341, 341)),
)
auto_resume = False
gpu_ids = range(0, 8)
work_dir = "./work_dirs/segmentation_eval/dinov2_vitb14_voc2012_linear"
