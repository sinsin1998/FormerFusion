seed = 0
deterministic = False
checkpoint_config = dict(interval=1, max_keep_ckpts=2)
log_config = dict(
    interval=50,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])
# load checkpoint
load_from = 'pretrained/lidar-only-det.pth'
resume_from = None
cudnn_benchmark = False
# float
fp16 = dict(loss_scale=dict(growth_interval=2000))
max_epochs = 24
# runner
runner = dict(type='CustomEpochBasedRunner', max_epochs=24)
dataset_type = 'NuScenesDataset'
dataset_root = '/data/nuscenes/'
gt_paste_stop_epoch = -1
reduce_beams = 32
load_dim = 5
use_dim = 5
load_augmented = None
# perception range
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
voxel_size = [0.1, 0.1, 0.2]
pts_point_cloud_range=[-54.0, -54.0, -5.0, 54.0, 54.0, 3.0],
pts_voxel_size=[0.075, 0.075, 0.2]
image_size = [256, 704]
augment2d = dict(
    resize=[[0.38, 0.55], [0.48, 0.48]],
    rotate=[-5.4, 5.4],
    gridmask=dict(prob=0.0, fixed_prob=True))
augment3d = dict(
    scale=[0.9, 1.1], rotate=[-0.78539816, 0.78539816], translate=0.5)
object_classes = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
input_modality = dict(
    use_lidar=True,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)
train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        reduce_beams=32,
        load_augmented=None),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=9,
        load_dim=5,
        use_dim=5,
        reduce_beams=32,
        pad_empty_sweeps=True,
        remove_close=True,
        load_augmented=None),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True,
        with_attr_label=False),
    dict(
        type='ObjectPaste',
        stop_epoch=-1,
        db_sampler=dict(
            dataset_root='/data/nuscenes/',
            info_path='/data/nuscenes/nuscenes_dbinfos_train.pkl',
            rate=1.0,
            prepare=dict(
                filter_by_difficulty=[-1],
                filter_by_min_points=dict(
                    car=5,
                    truck=5,
                    bus=5,
                    trailer=5,
                    construction_vehicle=5,
                    traffic_cone=5,
                    barrier=5,
                    motorcycle=5,
                    bicycle=5,
                    pedestrian=5)),
            classes=[
                'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                'barrier', 'motorcycle', 'bicycle', 'pedestrian',
                'traffic_cone'
            ],
            sample_groups=dict(
                car=2,
                truck=3,
                construction_vehicle=7,
                bus=4,
                trailer=6,
                barrier=2,
                motorcycle=6,
                bicycle=6,
                pedestrian=2,
                traffic_cone=2),
            points_loader=dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=5,
                use_dim=5,
                reduce_beams=32))),
    dict(
        type='ImageAug3D',
        final_dim=[256, 704],
        resize_lim=[0.38, 0.55],
        bot_pct_lim=[0.0, 0.0],
        rot_lim=[-5.4, 5.4],
        rand_flip=True,
        is_train=True),
    dict(type='RandomFlip3D'),
    dict(
        type='PointsRangeFilter',
        point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]),
    dict(
        type='ObjectRangeFilter',
        point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]),
    dict(
        type='ObjectNameFilter',
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ]),
    dict(
        type='ImageNormalize',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(
        type='GridMask',
        use_h=True,
        use_w=True,
        max_epoch=24,
        rotate=1,
        offset=False,
        ratio=0.5,
        mode=1,
        prob=0.0,
        fixed_prob=True),
    dict(type='PointShuffle'),
    dict(
        type='DefaultFormatBundle3D',
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ]),
    dict(
        type='Collect3D',
        keys=['img', 'points', 'gt_bboxes_3d', 'gt_labels_3d'],
        meta_keys=[
            'camera_intrinsics', 'camera2ego', 'lidar2ego', 'lidar2camera',
            'camera2lidar', 'lidar2image', 'img_aug_matrix', 'lidar_aug_matrix'
        ])
]
test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        reduce_beams=32,
        load_augmented=None),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=9,
        load_dim=5,
        use_dim=5,
        reduce_beams=32,
        pad_empty_sweeps=True,
        remove_close=True,
        load_augmented=None),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True,
        with_attr_label=False),
    dict(
        type='ImageAug3D',
        final_dim=[256, 704],
        resize_lim=[0.48, 0.48],
        bot_pct_lim=[0.0, 0.0],
        rot_lim=[0.0, 0.0],
        rand_flip=False,
        is_train=False),
    dict(
        type='GlobalRotScaleTrans',
        resize_lim=[1.0, 1.0],
        rot_lim=[0.0, 0.0],
        trans_lim=0.0,
        is_train=False),
    dict(
        type='PointsRangeFilter',
        point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]),
    dict(
        type='ImageNormalize',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(
        type='DefaultFormatBundle3D',
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ]),
    dict(
        type='Collect3D',
        keys=['img', 'points', 'gt_bboxes_3d', 'gt_labels_3d'],
        meta_keys=[
            'camera_intrinsics', 'camera2ego', 'lidar2ego', 'lidar2camera',
            'camera2lidar', 'lidar2image', 'img_aug_matrix', 'lidar_aug_matrix'
        ])
]
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        type='CBGSDataset',
        dataset=dict(
            type='NuScenesDataset',
            dataset_root='/data/nuscenes/',
            ann_file='/data/nuscenes/nuscenes_infos_train.pkl',
            pipeline=[
                dict(type='LoadMultiViewImageFromFiles', to_float32=True),
                dict(
                    type='LoadPointsFromFile',
                    coord_type='LIDAR',
                    load_dim=5,
                    use_dim=5,
                    reduce_beams=32,
                    load_augmented=None),
                dict(
                    type='LoadPointsFromMultiSweeps',
                    sweeps_num=9,
                    load_dim=5,
                    use_dim=5,
                    reduce_beams=32,
                    pad_empty_sweeps=True,
                    remove_close=True,
                    load_augmented=None),
                dict(
                    type='LoadAnnotations3D',
                    with_bbox_3d=True,
                    with_label_3d=True,
                    with_attr_label=False),
                dict(
                    type='ObjectPaste',
                    stop_epoch=-1,
                    db_sampler=dict(
                        dataset_root='/data/nuscenes/',
                        info_path=
                        '/data/nuscenes/nuscenes_dbinfos_train.pkl',
                        rate=1.0,
                        prepare=dict(
                            filter_by_difficulty=[-1],
                            filter_by_min_points=dict(
                                car=5,
                                truck=5,
                                bus=5,
                                trailer=5,
                                construction_vehicle=5,
                                traffic_cone=5,
                                barrier=5,
                                motorcycle=5,
                                bicycle=5,
                                pedestrian=5)),
                        classes=[
                            'car', 'truck', 'construction_vehicle', 'bus',
                            'trailer', 'barrier', 'motorcycle', 'bicycle',
                            'pedestrian', 'traffic_cone'
                        ],
                        sample_groups=dict(
                            car=2,
                            truck=3,
                            construction_vehicle=7,
                            bus=4,
                            trailer=6,
                            barrier=2,
                            motorcycle=6,
                            bicycle=6,
                            pedestrian=2,
                            traffic_cone=2),
                        points_loader=dict(
                            type='LoadPointsFromFile',
                            coord_type='LIDAR',
                            load_dim=5,
                            use_dim=5,
                            reduce_beams=32))),
                dict(
                    type='ImageAug3D',
                    final_dim=[256, 704],
                    resize_lim=[0.38, 0.55],
                    bot_pct_lim=[0.0, 0.0],
                    rot_lim=[-5.4, 5.4],
                    rand_flip=True,
                    is_train=True),
                dict(
                    type='GlobalRotScaleTrans',
                    resize_lim=[0.9, 1.1],
                    rot_lim=[-0.78539816, 0.78539816],
                    trans_lim=0.5,
                    is_train=True),
                dict(type='RandomFlip3D'),
                dict(
                    type='PointsRangeFilter',
                    point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]),
                dict(
                    type='ObjectRangeFilter',
                    point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]),
                dict(
                    type='ObjectNameFilter',
                    classes=[
                        'car', 'truck', 'construction_vehicle', 'bus',
                        'trailer', 'barrier', 'motorcycle', 'bicycle',
                        'pedestrian', 'traffic_cone'
                    ]),
                dict(
                    type='ImageNormalize',
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
                dict(
                    type='GridMask',
                    use_h=True,
                    use_w=True,
                    max_epoch=24,
                    rotate=1,
                    offset=False,
                    ratio=0.5,
                    mode=1,
                    prob=0.0,
                    fixed_prob=True),
                dict(type='PointShuffle'),
                dict(
                    type='DefaultFormatBundle3D',
                    classes=[
                        'car', 'truck', 'construction_vehicle', 'bus',
                        'trailer', 'barrier', 'motorcycle', 'bicycle',
                        'pedestrian', 'traffic_cone'
                    ]),
                dict(
                    type='Collect3D',
                    keys=[
                        'img', 'points', 'gt_bboxes_3d', 'gt_labels_3d',
                        'gt_masks_bev'
                    ],
                    meta_keys=[
                        'camera_intrinsics', 'camera2ego', 'lidar2ego',
                        'lidar2camera', 'camera2lidar', 'lidar2image',
                        'img_aug_matrix', 'lidar_aug_matrix'
                    ])
            ],
            object_classes=[
                'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                'barrier', 'motorcycle', 'bicycle', 'pedestrian',
                'traffic_cone'
            ],
            modality=dict(
                use_lidar=True,
                use_camera=True,
                use_radar=False,
                use_map=False,
                use_external=False),
            test_mode=False,
            use_valid_flag=True,
            box_type_3d='LiDAR')),
    val=dict(
        type='NuScenesDataset',
        dataset_root='/data/nuscenes/',
        ann_file='/data/nuscenes/nuscenes_infos_val.pkl',
        pipeline=[
            dict(type='LoadMultiViewImageFromFiles', to_float32=True),
            dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=5,
                use_dim=5,
                reduce_beams=32,
                load_augmented=None),
            dict(
                type='LoadPointsFromMultiSweeps',
                sweeps_num=9,
                load_dim=5,
                use_dim=5,
                reduce_beams=32,
                pad_empty_sweeps=True,
                remove_close=True,
                load_augmented=None),
            dict(
                type='LoadAnnotations3D',
                with_bbox_3d=True,
                with_label_3d=True,
                with_attr_label=False),
            dict(
                type='ImageAug3D',
                final_dim=[256, 704],
                resize_lim=[0.48, 0.48],
                bot_pct_lim=[0.0, 0.0],
                rot_lim=[0.0, 0.0],
                rand_flip=False,
                is_train=False),
            dict(
                type='GlobalRotScaleTrans',
                resize_lim=[1.0, 1.0],
                rot_lim=[0.0, 0.0],
                trans_lim=0.0,
                is_train=False),
            dict(
                type='PointsRangeFilter',
                point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]),
            dict(
                type='ImageNormalize',
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
            dict(
                type='DefaultFormatBundle3D',
                classes=[
                    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                    'barrier', 'motorcycle', 'bicycle', 'pedestrian',
                    'traffic_cone'
                ]),
            dict(
                type='Collect3D',
                keys=[
                    'img', 'points', 'gt_bboxes_3d', 'gt_labels_3d',
                    'gt_masks_bev'
                ],
                meta_keys=[
                    'camera_intrinsics', 'camera2ego', 'lidar2ego',
                    'lidar2camera', 'camera2lidar', 'lidar2image',
                    'img_aug_matrix', 'lidar_aug_matrix'
                ])
        ],
        object_classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ],
        modality=dict(
            use_lidar=True,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=False),
        test_mode=False,
        box_type_3d='LiDAR'),
    test=dict(
        type='NuScenesDataset',
        dataset_root='/data/nuscenes/',
        ann_file='/data/nuscenes/nuscenes_infos_val.pkl',
        pipeline=[
            dict(type='LoadMultiViewImageFromFiles', to_float32=True),
            dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=5,
                use_dim=5,
                reduce_beams=32,
                load_augmented=None),
            dict(
                type='LoadPointsFromMultiSweeps',
                sweeps_num=9,
                load_dim=5,
                use_dim=5,
                reduce_beams=32,
                pad_empty_sweeps=True,
                remove_close=True,
                load_augmented=None),
            dict(
                type='LoadAnnotations3D',
                with_bbox_3d=True,
                with_label_3d=True,
                with_attr_label=False),
            dict(
                type='ImageAug3D',
                final_dim=[256, 704],
                resize_lim=[0.48, 0.48],
                bot_pct_lim=[0.0, 0.0],
                rot_lim=[0.0, 0.0],
                rand_flip=False,
                is_train=False),
            dict(
                type='GlobalRotScaleTrans',
                resize_lim=[1.0, 1.0],
                rot_lim=[0.0, 0.0],
                trans_lim=0.0,
                is_train=False),
            dict(
                type='PointsRangeFilter',
                point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]),
            dict(
                type='ImageNormalize',
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
            dict(
                type='DefaultFormatBundle3D',
                classes=[
                    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                    'barrier', 'motorcycle', 'bicycle', 'pedestrian',
                    'traffic_cone'
                ]),
            dict(
                type='Collect3D',
                keys=[
                    'img', 'points', 'gt_bboxes_3d', 'gt_labels_3d',
                    'gt_masks_bev'
                ],
                meta_keys=[
                    'camera_intrinsics', 'camera2ego', 'lidar2ego',
                    'lidar2camera', 'camera2lidar', 'lidar2image',
                    'img_aug_matrix', 'lidar_aug_matrix'
                ])
        ],
        object_classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ],
        modality=dict(
            use_lidar=True,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=False),
        test_mode=True,
        box_type_3d='LiDAR'))
evaluation = dict(interval=1, pipeline=test_pipeline)
model = dict(
    type='FormFusion',
    encoders=dict(
        camera=dict(
            neck=dict(
                type='GeneralizedLSSFPN',
                in_channels=[192, 384, 768],
                out_channels=256,
                start_level=0,
                num_outs=3,
                norm_cfg=dict(type='BN2d', requires_grad=True),
                act_cfg=dict(type='ReLU', inplace=True),
                upsample_cfg=dict(mode='bilinear', align_corners=False)),
            backbone=dict(
                type='SwinTransformer',
                embed_dims=96,
                depths=[2, 2, 6, 2],
                num_heads=[3, 6, 12, 24],
                window_size=7,
                mlp_ratio=4,
                qkv_bias=True,
                qk_scale=None,
                drop_rate=0.0,
                attn_drop_rate=0.0,
                drop_path_rate=0.2,
                patch_norm=True,
                out_indices=[1, 2, 3],
                with_cp=False,
                convert_weights=True,
                init_cfg=dict(
                    type='Pretrained',
                    checkpoint='pretrained/swint-nuimages-pretrained.pth'))),
        lidar=dict(
            voxelize=dict(
                max_num_points=10,
                point_cloud_range=[-54.0, -54.0, -5.0, 54.0, 54.0, 3.0],
                voxel_size=[0.075, 0.075, 0.2],
                #max_voxels=[90000, 120000],
                max_voxels=[120000, 160000],
                ),
            backbone=dict(
                type='SparseEncoder',
                in_channels=5,
                #sparse_shape=[1024, 1024, 41],
                sparse_shape=[1440, 1440, 41],
                output_channels=128,
                order=['conv', 'norm', 'act'],
                encoder_channels=[[16, 16, 32], [32, 32, 64], [64, 64, 128],
                                  [128, 128]],
                encoder_paddings=[[0, 0, 1], [0, 0, 1], [0, 0, [1, 1, 0]],
                                  [0, 0]],
                block_type='basicblock'))),
    fuser=None, 
    heads=dict( 
        object=dict(
            type='FormfusionHead',
            _dim_=256,
            num_query=900,
            num_classes=10,
            sync_cls_avg_factor=True,
            with_box_refine=True,
            as_two_stage=False,
            in_channels=[80, 256],
            transformer=dict(
                type='FormerfusionTransformer',
                rotate_prev_bev=False,
                use_shift=False,
                use_can_bus=False,
                embed_dims=256,
                encoder=dict(
                    type='FormerfusionEncoder',
                    num_layers=6,
                    pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
                    num_points_in_pillar=4,
                    return_intermediate=False,
                    transformerlayers=dict(
                        type='FormerfusionLayer',
                        attn_cfgs=[
                            dict(
                                type='TemporalSelfAttention',
                                embed_dims=256,
                                num_levels=1),
                            dict(
                                type='SpatialCrossAttention',
                                pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
                                deformable_attention=dict(
                                    type='MSDeformableAttention3D',
                                    embed_dims=256,
                                    num_points=8,
                                    num_levels=1),
                                embed_dims=256)
                        ],
                        feedforward_channels=512,
                        ffn_dropout=0.1,
                        operation_order=[
                            'self_attn', 'norm', 'cross_attn', 'norm', 'ffn',
                            'norm'
                        ])),
                decoder=dict(
                    type='DetectionTransformerDecoder',
                    num_layers=6,
                    return_intermediate=True,
                    transformerlayers=dict(
                        type='DetrTransformerDecoderLayer',
                        attn_cfgs=[
                            dict(
                                type='MultiheadAttention',
                                embed_dims=256,
                                num_heads=8,
                                dropout=0.1),
                            dict(
                                type='CustomMSDeformableAttention',
                                embed_dims=256,
                                num_levels=1)
                        ],
                        feedforward_channels=512,
                        ffn_dropout=0.1,
                        operation_order=[
                            'self_attn', 'norm', 'cross_attn', 'norm', 'ffn',
                            'norm'
                        ]))),
            bbox_coder=dict(
                type='NMSFreeCoder',
                post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
                pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
                max_num=300,
                voxel_size=[0.2, 0.2, 8],
                num_classes=10),
            positional_encoding=dict(
                type='LearnedPositionalEncoding',
                num_feats=128,
                row_num_embed=180,
                col_num_embed=180),
            loss_cls=dict(
                type='FocalLoss',
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                reduction='mean',
                loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', reduction='mean', loss_weight=0.25),
            loss_iou=dict(type='GIoULoss', loss_weight=0),
            train_cfg=dict(
                assigner=dict(
                    type='HungarianAssigner3D',
                    iou_calculator=dict(
                        type='BboxOverlaps3D', coordinate='lidar'),
                    cls_cost=dict(
                        type='FocalLossCost',
                        gamma=2.0,
                        alpha=0.25,
                        weight=1.0),
                    reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
                    iou_cost=dict(type='IoUCost', weight=0))),
            )),
    train_cfg=dict(
        pts=dict(
            grid_size=[1024, 1024, 1], # change from [512,512,1]
            voxel_size=[0.075, 0.075, 0.2],
            point_cloud_range=[-54.0, -54.0, -5.0, 54.0, 54.0, 3.0],
            out_size_factor=4,
            assigner=dict(
                type='HungarianAssigner3D',
                iou_calculator=dict(type='BboxOverlaps3D', coordinate='lidar'),
                cls_cost=dict(
                    type='FocalLossCost', gamma=2.0, alpha=0.25, weight=1.0),
                reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
                iou_cost=dict(type='IoUCost', weight=0)))),
    decoder=None 
    )
_dim_ = 256
optimizer = dict(type='AdamW', lr=0.0002, weight_decay=0.01)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.33333333,
    min_lr_ratio=0.001)
momentum_config = dict(policy='cyclic')
