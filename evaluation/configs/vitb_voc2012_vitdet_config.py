_base_ = [
    "./models/faster_rcnn_r50_fpn.py",
]
custom_imports = dict(imports=["evaluation.vitdet"])

dataset_type = "VOCDataset"
data_root = "data/VOCdevkit/"
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(type="Resize", img_scale=(512, 512), keep_ratio=True),
    dict(type="RandomFlip", flip_ratio=0.5),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="Pad", size_divisor=32),
    dict(type="DefaultFormatBundle"),
    dict(type="Collect", keys=["img", "gt_bboxes", "gt_labels"]),
]
test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="MultiScaleFlipAug",
        img_scale=(512, 512),
        flip=False,
        transforms=[
            dict(type="Resize", keep_ratio=True),
            dict(type="RandomFlip"),
            dict(type="Normalize", **img_norm_cfg),
            dict(type="Pad", size_divisor=32),
            dict(type="ImageToTensor", keys=["img"]),
            dict(type="Collect", keys=["img"]),
        ],
    ),
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=[
            data_root + "VOC2007/ImageSets/Main/trainval.txt",
            data_root + "VOC2012/ImageSets/Main/trainval.txt",
        ],
        img_prefix=[data_root + "VOC2007/", data_root + "VOC2012/"],
        pipeline=train_pipeline,
    ),
    val=dict(
        type=dataset_type,
        ann_file=data_root + "VOC2007/ImageSets/Main/test.txt",
        img_prefix=data_root + "VOC2007/",
        pipeline=test_pipeline,
    ),
    test=dict(
        type=dataset_type,
        ann_file=data_root + "VOC2007/ImageSets/Main/test.txt",
        img_prefix=data_root + "VOC2007/",
        pipeline=test_pipeline,
    ),
)

norm_cfg = dict(type="LN2d", requires_grad=True)
model = dict(
    backbone=dict(type="DinoVisionTransformer", out_indices=[11]),
    neck=dict(
        _delete_=True,
        type="SimpleFPN",
        backbone_channel=768,
        in_channels=[192, 384, 768, 768],
        out_channels=256,
        num_outs=5,
        norm_cfg=norm_cfg,
    ),
    rpn_head=dict(num_convs=2),
    roi_head=dict(
        bbox_head=dict(
            type="Shared4Conv1FCBBoxHead",
            conv_out_channels=256,
            norm_cfg=norm_cfg,
            num_classes=20,
        ),
    ),
)

log_config = dict(interval=50, hooks=[dict(type="TextLoggerHook", by_epoch=False)])
custom_hooks = [dict(type="NumClassCheckHook"), dict(type="Fp16CompresssionHook")]
dist_params = dict(backend="nccl")
log_level = "INFO"
load_from = None
resume_from = None
workflow = [("train", 1)]
cudnn_benchmark = True

optimizer = dict(
    type="AdamW",
    lr=0.0001,
    betas=(0.9, 0.999),
    weight_decay=0.1,
)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy="step",
    warmup="linear",
    warmup_iters=100,
    warmup_ratio=0.001,
    by_epoch=False,
    gamma=0.1,
    step=[20000, 22000],
)

# Runner type
runner = dict(type="IterBasedRunner", max_iters=24000)
checkpoint_config = dict(interval=4000, save_last=True, max_keep_ckpts=5)
evaluation = dict(interval=4000, metric="mAP")
auto_scale_lr = dict(base_batch_size=16)
