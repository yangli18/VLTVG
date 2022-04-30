from .dataset import VGDataset


def build_dataset(test, args):
    if test:
        return VGDataset(data_root=args.data_root,
                         split_root=args.split_root,
                         dataset=args.dataset,
                         split=args.test_split,
                         test=True,
                         transforms=args.test_transforms,
                         max_query_len=args.max_query_len,
                         bert_mode=args.bert_token_mode)
    else:
        return VGDataset(data_root=args.data_root,
                          split_root=args.split_root,
                          dataset=args.dataset,
                          split='train',
                          transforms=args.train_transforms,
                          max_query_len=args.max_query_len,
                          bert_mode=args.bert_token_mode)


train_transforms = [
    dict(
        type='RandomSelect',
        transforms1=dict(type='RandomResize', sizes=[448, 480, 512, 544, 576, 608, 640]),
        transforms2=dict(
            type='Compose',
            transforms=[
                dict(type='RandomResize', sizes=[400, 500, 600], resize_long_side=False),
                dict(type='RandomSizeCrop', min_size=384, max_size=600, check_method=dict(func='iou', iou_thres=0.5)),
                dict(type='RandomResize', sizes=[448, 480, 512, 544, 576, 608, 640])
            ],
        ),
        p=0.5
    ),
    dict(type='ColorJitter', brightness=0.4, contrast=0.4, saturation=0.4),
    dict(type='RandomHorizontalFlip'),
    dict(type='ToTensor', keys=[]),
    dict(type='NormalizeAndPad', size=640, aug_translate=True)
]

test_transforms = [
    dict(type='RandomResize', sizes=[640], record_resize_info=True),
    dict(type='ToTensor', keys=[]),
    dict(type='NormalizeAndPad', size=640, center_place=True)
]