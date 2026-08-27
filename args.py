import argparse

def get_parser():
    parser = argparse.ArgumentParser(description='FIANet training and testing')
    parser.add_argument('--amsgrad', action='store_true',
                        help='if true, set amsgrad to True in an Adam or AdamW optimizer.')
    parser.add_argument('-b', '--batch-size', default=8, type=int)
    parser.add_argument('--bert_tokenizer', default='./bert-base-uncased', help='BERT tokenizer')
    parser.add_argument('--ck_bert', default='bert-base-uncased', help='pre-trained BERT weights')
    parser.add_argument('--dataset', default='rrsisd', help='refcoco, refcoco+, or refcocog')
    parser.add_argument('--ddp_trained_weights', action='store_true',
                        help='Only needs specified when testing,'
                             'whether the weights to be loaded are from a DDP-trained model')
    parser.add_argument('--device', default='cuda:0', help='device')  # only used when testing on a single machine
    parser.add_argument('--epochs', default=60, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--fusion_drop', default=0.0, type=float, help='dropout rate for PWAMs')
    parser.add_argument('--img_size', default=480, type=int, help='input image size')
    parser.add_argument("--local_rank", type=int,default=0,help='local rank for DistributedDataParallel')
    parser.add_argument('--lr', default=5e-5, type=float, help='the initial learning rate')   # 5e-5 for RefSegRS, 3e-5 for RRSIS-D
    parser.add_argument('--mha', default='', help='If specified, should be in the format of a-b-c-d, e.g., 4-4-4-4,'
                                                  'where a, b, c, and d refer to the numbers of heads in stage-1,'
                                                  'stage-2, stage-3, and stage-4 PWAMs')
    parser.add_argument('--model', default='lavt_one', help='model: lavt, lavt_one')
    parser.add_argument('--model_id', default='FIANet', help='name to identify the model')
    parser.add_argument('--output-dir', default='./checkpoints/', help='path where to save checkpoint weights')
    parser.add_argument('--pin_mem', action='store_true',
                        help='If true, pin memory when using the data loader.')
    parser.add_argument('--pretrained_swin_weights', default='./pretrained_weights/swin_base_patch4_window12_384_22k.pth',
                        help='path to pre-trained Swin backbone weights')
    parser.add_argument('--print-freq', default=10, type=int, help='print frequency')
    parser.add_argument('--refer_data_root', default='C:/Dataset/refer_seg/RefSegRS/', help='REFER dataset root directory')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--split', default='test', help='only used when testing')
    parser.add_argument('--splitBy', default='unc', help='change to umd or google when the datasset is G-Ref (RefCOCOg)')
    parser.add_argument('--swin_type', default='base',
                        help='tiny, small, base, or large variants of the Swin Transformer')
    parser.add_argument('--wd', '--weight-decay', default=1e-2, type=float, metavar='W', help='weight decay',
                        dest='weight_decay')
    parser.add_argument('--window12', action='store_true',
                        help='only needs specified when testing,'
                             'when training, window size is inferred from pre-trained weights file name'
                             '(containing \'window12\'). Initialize Swin with window size 12 instead of the default 7.')
    parser.add_argument('-j', '--workers', default=0, type=int, metavar='N', help='number of data loading workers')
    parser.add_argument('--num_tmem', default=1, type=int, help='number of tmem layers')  # 1 for RefSegRS, 3 for RRSIS-D
    # E1 auxiliary coarse localization supervision.
    parser.add_argument(
        '--use-coarse-loc',
        action='store_true',
        help='enable the E1 coarse localization head',
    )

    parser.add_argument(
        '--aux-loc-weight',
        default=0.3,
        type=float,
        help='weight of the E1 auxiliary localization loss',
    )

    # E2 prompt-guided residual refinement.
    parser.add_argument(
        '--use-prompt-refine',
        action='store_true',
        help='enable E2 prompt-guided residual refinement',
    )

    parser.add_argument(
        '--refine-channels',
        default=32,
        type=int,
        help='hidden channels of the E2 refinement head',
    )

    parser.add_argument(
        '--prompt-channels',
        default=16,
        type=int,
        help='prompt embedding channels of the E2 head',
    )

    # S2 high-resolution small-object refinement.
    parser.add_argument(
        '--use-small-refine',
        action='store_true',
        help=(
            'enable the S2 high-resolution '
            'small-object residual refinement head'
        ),
    )

    parser.add_argument(
        '--small-project-channels',
        default=16,
        type=int,
        help=(
            'projection channels for each '
            'S2 high-resolution feature'
        ),
    )

    parser.add_argument(
        '--small-refine-channels',
        default=32,
        type=int,
        help='hidden channels of the S2 refinement head',
    )

    # S1-A: size-aware training-sample resampling.
    parser.add_argument(
        '--use-size-aware-sampler',
        action='store_true',
        help=(
            'enable S1-A weighted replacement sampling '
            'according to GT area ratio'
        ),
    )

    parser.add_argument(
        '--s1a-tiny-weight',
        default=2.0,
        type=float,
        help='sampling weight for Tiny targets',
    )

    parser.add_argument(
        '--s1a-small-weight',
        default=1.5,
        type=float,
        help='sampling weight for Small targets',
    )

    parser.add_argument(
        '--s1a-medium-weight',
        default=1.0,
        type=float,
        help='sampling weight for Medium targets',
    )

    parser.add_argument(
        '--s1a-large-weight',
        default=1.0,
        type=float,
        help='sampling weight for Large targets',
    )

    parser.add_argument(
        '--s1a-empty-weight',
        default=1.0,
        type=float,
        help='sampling weight for empty-GT samples',
    )

    parser.add_argument(
        '--s1a-tiny-max-ratio',
        default=0.001,
        type=float,
        help='maximum foreground ratio for Tiny',
    )

    parser.add_argument(
        '--s1a-small-max-ratio',
        default=0.005,
        type=float,
        help='maximum foreground ratio for Small',
    )

    parser.add_argument(
        '--s1a-medium-max-ratio',
        default=0.020,
        type=float,
        help='maximum foreground ratio for Medium',
    )

    parser.add_argument(
        '--s1a-sampler-seed',
        default=2401,
        type=int,
        help='random seed used by the S1-A sampler',
    )

    parser.add_argument(
        '--s1a-area-cache',
        default=(
            'experiments/S1A/'
            'rrsisd_train_area_ratios.json'
        ),
        type=str,
        help='cache of training GT area ratios',
    )

    parser.add_argument(
        '--s1a-audit-output',
        default=(
            'experiments/S1A/'
            's1a_sampler_audit.json'
        ),
        type=str,
        help='output JSON for sampler audit',
    )

    # S1-B: size-aware sample-level loss weighting.
    parser.add_argument(
        '--use-size-aware-loss',
        action='store_true',
        help=(
            'enable S1-B size-aware sample-level '
            'segmentation loss weighting'
        ),
    )

    parser.add_argument(
        '--s1b-tiny-weight',
        default=1.5,
        type=float,
        help='S1-B loss weight for Tiny targets',
    )

    parser.add_argument(
        '--s1b-small-weight',
        default=1.25,
        type=float,
        help='S1-B loss weight for Small targets',
    )

    parser.add_argument(
        '--s1b-medium-weight',
        default=1.0,
        type=float,
        help='S1-B loss weight for Medium targets',
    )

    parser.add_argument(
        '--s1b-large-weight',
        default=1.0,
        type=float,
        help='S1-B loss weight for Large targets',
    )

    parser.add_argument(
        '--s1b-empty-weight',
        default=1.0,
        type=float,
        help='S1-B loss weight for empty targets',
    )

    parser.add_argument(
        '--s1b-tiny-max-ratio',
        default=0.001,
        type=float,
        help='maximum foreground ratio for Tiny',
    )

    parser.add_argument(
        '--s1b-small-max-ratio',
        default=0.005,
        type=float,
        help='maximum foreground ratio for Small',
    )

    parser.add_argument(
        '--s1b-medium-max-ratio',
        default=0.020,
        type=float,
        help='maximum foreground ratio for Medium',
    )

    parser.add_argument(
        '--s1b-dice-mix-weight',
        default=0.1,
        type=float,
        help=(
            'Dice mixture coefficient; '
            'the E0 value is 0.1'
        ),
    )

    parser.add_argument(
        '--s1b-log-first-batch',
        action='store_true',
        help=(
            'print S1-B group assignments and '
            'loss components for the first batch'
        ),
    )

    # S1-C: Tiny/Small foreground-only Dice auxiliary loss.
    parser.add_argument(
        '--use-foreground-size-aux-loss',
        action='store_true',
        help=(
            'enable S1-C Tiny/Small foreground-only '
            'Dice auxiliary loss'
        ),
    )

    parser.add_argument(
        '--s1c-tiny-lambda',
        default=0.30,
        type=float,
        help=(
            'foreground Dice auxiliary coefficient '
            'for Tiny samples'
        ),
    )

    parser.add_argument(
        '--s1c-small-lambda',
        default=0.15,
        type=float,
        help=(
            'foreground Dice auxiliary coefficient '
            'for Small samples'
        ),
    )

    parser.add_argument(
        '--s1c-tiny-max-ratio',
        default=0.001,
        type=float,
        help='maximum foreground ratio for Tiny',
    )

    parser.add_argument(
        '--s1c-small-max-ratio',
        default=0.005,
        type=float,
        help='maximum foreground ratio for Small',
    )

    parser.add_argument(
        '--s1c-log-first-batch',
        action='store_true',
        help=(
            'print S1-C first-batch areas, '
            'coefficients and loss components'
        ),
    )

    # S1-Cv2 scheduled foreground Dice.
    parser.add_argument(
        '--use-scheduled-foreground-size-aux-loss',
        action='store_true',
    )

    parser.add_argument(
        '--s1cv2-tiny-max-lambda',
        default=0.10,
        type=float,
    )

    parser.add_argument(
        '--s1cv2-small-max-lambda',
        default=0.05,
        type=float,
    )

    parser.add_argument(
        '--s1cv2-tiny-max-ratio',
        default=0.001,
        type=float,
    )

    parser.add_argument(
        '--s1cv2-small-max-ratio',
        default=0.005,
        type=float,
    )

    parser.add_argument(
        '--s1cv2-warmup-epochs',
        default=5,
        type=int,
    )

    parser.add_argument(
        '--s1cv2-ramp-epochs',
        default=10,
        type=int,
    )

    parser.add_argument(
        '--s1cv2-hold-epochs',
        default=16,
        type=int,
    )

    parser.add_argument(
        '--s1cv2-decay-epochs',
        default=9,
        type=int,
    )

    parser.add_argument(
        '--s1cv2-pilot-epochs',
        default=0,
        type=int,
        help=(
            'number of epochs actually executed; '
            '0 runs all --epochs'
        ),
    )

    # A1: original-GT versus resized-GT size audit.
    parser.add_argument(
        '--a1-split',
        default='test',
        choices=('train', 'val', 'test'),
    )

    parser.add_argument(
        '--a1-output-dir',
        default='experiments/A1',
    )

    parser.add_argument(
        '--a1-tag',
        default='RRSISD_test',
    )

    parser.add_argument(
        '--a1-tiny-max-ratio',
        default=0.001,
        type=float,
    )

    parser.add_argument(
        '--a1-small-max-ratio',
        default=0.005,
        type=float,
    )

    parser.add_argument(
        '--a1-medium-max-ratio',
        default=0.02,
        type=float,
    )

    parser.add_argument(
        '--a1-low-mapping-iou',
        default=0.90,
        type=float,
    )

    parser.add_argument(
        '--a1-print-freq',
        default=200,
        type=int,
    )

    # S3-A: local positive and hard-negative ring loss.
    parser.add_argument(
        '--use-s3a-local-ring',
        action='store_true',
        help=(
            'enable S3-A local positive and '
            'hard-negative ring supervision'
        ),
    )

    parser.add_argument(
        '--s3a-positive-weight',
        default=0.05,
        type=float,
    )

    parser.add_argument(
        '--s3a-ring-weight',
        default=0.05,
        type=float,
    )

    parser.add_argument(
        '--s3a-ring-radius',
        default=8,
        type=int,
    )

    parser.add_argument(
        '--s3a-tiny-max-ratio',
        default=0.001,
        type=float,
    )

    parser.add_argument(
        '--s3a-small-max-ratio',
        default=0.005,
        type=float,
    )

    parser.add_argument(
        '--s3a-warmup-epochs',
        default=5,
        type=int,
    )

    parser.add_argument(
        '--s3a-ramp-epochs',
        default=5,
        type=int,
    )

    return parser


if __name__ == "__main__":
    parser = get_parser()
    args_dict = parser.parse_args()
