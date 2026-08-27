import torch
import torch.nn as nn
from .mask_predictor import SimpleDecoding
from .coarse_localization_head import CoarseLocalizationHead
from .prompt_guided_refinement import PromptGuidedResidualRefinementHead
from .high_resolution_small_refinement import HighResolutionSmallObjectRefinementHead
from .backbone import MultiModalSwinTransformer
from ._utils import LAVT, LAVTOne


__all__ = ['lavt', 'lavt_one']


# LAVT
def _segm_lavt(pretrained, args):
    # initialize the SwinTransformer backbone with the specified version
    if args.swin_type == 'tiny':
        embed_dim = 96
        depths = [2, 2, 6, 2]
        num_heads = [3, 6, 12, 24]
    elif args.swin_type == 'small':
        embed_dim = 96
        depths = [2, 2, 18, 2]
        num_heads = [3, 6, 12, 24]
    elif args.swin_type == 'base':
        embed_dim = 128
        depths = [2, 2, 18, 2]
        num_heads = [4, 8, 16, 32]
    elif args.swin_type == 'large':
        embed_dim = 192
        depths = [2, 2, 18, 2]
        num_heads = [6, 12, 24, 48]
    else:
        assert False
    # args.window12 added for test.py because state_dict is loaded after model initialization
    if 'window12' in pretrained or args.window12:
        print('Window size 12!')
        window_size = 12
    else:
        window_size = 7

    if args.mha:
        mha = args.mha.split('-')  # if non-empty, then ['a', 'b', 'c', 'd']
        mha = [int(a) for a in mha]
    else:
        mha = [1, 1, 1, 1]

    out_indices = (0, 1, 2, 3)
    backbone = MultiModalSwinTransformer(embed_dim=embed_dim, depths=depths, num_heads=num_heads,
                                         window_size=window_size,
                                         num_tmem=args.num_tmem,
                                         ape=False, drop_path_rate=0.3, patch_norm=True,
                                         out_indices=out_indices,
                                         use_checkpoint=False, num_heads_fusion=mha,
                                         fusion_drop=args.fusion_drop
                                         )
    if pretrained:
        print('Initializing Multi-modal Swin Transformer weights from ' + pretrained)
        backbone.init_weights(pretrained=pretrained)
    else:
        print('Randomly initialize Multi-modal Swin Transformer weights.')
        backbone.init_weights()

    model_map = [SimpleDecoding, LAVT]

    classifier = model_map[0](8*embed_dim)
    base_model = model_map[1]

    model = base_model(backbone, classifier)
    return model


def _load_model_lavt(pretrained, args):
    model = _segm_lavt(pretrained, args)
    return model


def lavt(pretrained='', args=None):
    return _load_model_lavt(pretrained, args)


###############################################
# LAVT One: put BERT inside the overall model #
###############################################
def _segm_lavt_one(pretrained, args):
    # initialize the SwinTransformer backbone with the specified version
    if args.swin_type == 'tiny':
        embed_dim = 96
        depths = [2, 2, 6, 2]
        num_heads = [3, 6, 12, 24]
    elif args.swin_type == 'small':
        embed_dim = 96
        depths = [2, 2, 18, 2]
        num_heads = [3, 6, 12, 24]
    elif args.swin_type == 'base':
        embed_dim = 128
        depths = [2, 2, 18, 2]
        num_heads = [4, 8, 16, 32]
    elif args.swin_type == 'large':
        embed_dim = 192
        depths = [2, 2, 18, 2]
        num_heads = [6, 12, 24, 48]
    else:
        assert False
    # args.window12 added for test.py because state_dict is loaded after model initialization
    if 'window12' in pretrained or args.window12:
        print('Window size 12!')
        window_size = 12
    else:
        window_size = 7

    if args.mha:
        mha = args.mha.split('-')  # if non-empty, then ['a', 'b', 'c', 'd']
        mha = [int(a) for a in mha]
    else:
        mha = [1, 1, 1, 1]

    out_indices = (0, 1, 2, 3)
    backbone = MultiModalSwinTransformer(embed_dim=embed_dim, depths=depths, num_heads=num_heads,
                                         window_size=window_size,
                                         num_tmem=args.num_tmem,
                                         ape=False, drop_path_rate=0.3, patch_norm=True,
                                         out_indices=out_indices,
                                         use_checkpoint=False, num_heads_fusion=mha,
                                         fusion_drop=args.fusion_drop,
                                         )
    if pretrained:
        print('Initializing Multi-modal Swin Transformer weights from ' + pretrained)
        backbone.init_weights(pretrained=pretrained)
    else:
        print('Randomly initialize Multi-modal Swin Transformer weights.')
        backbone.init_weights()

    model_map = [SimpleDecoding, LAVTOne]
    classifier = model_map[0](8*embed_dim)
    base_model = model_map[1]

    model = base_model(
        backbone,
        classifier,
        args,
    )

    if getattr(args, 'use_coarse_loc', False):
        model.coarse_head = CoarseLocalizationHead(
            in_channels=8 * embed_dim,
        )

        print(
            'E1 coarse localization head enabled: '
            'in_channels={}, aux_weight={}'.format(
                8 * embed_dim,
                args.aux_loc_weight,
            )
        )

    if getattr(args, 'use_prompt_refine', False):
        if model.coarse_head is None:
            raise ValueError(
                "--use-prompt-refine requires "
                "--use-coarse-loc"
            )

        # x_c2 channels = 2 * embed_dim.
        model.refinement_head = (
            PromptGuidedResidualRefinementHead(
                in_channels=2 * embed_dim,
                hidden_channels=args.refine_channels,
                prompt_channels=args.prompt_channels,
            )
        )

        print(
            "E2 prompt-guided residual refinement enabled: "
            "in_channels={}, hidden={}, prompt={}".format(
                2 * embed_dim,
                args.refine_channels,
                args.prompt_channels,
            )
        )

    if getattr(args, 'use_small_refine', False):
        # Keep S2 as a clean E0-based ablation.
        if (
            model.coarse_head is not None
            or model.refinement_head is not None
        ):
            raise ValueError(
                "--use-small-refine must not be combined "
                "with --use-coarse-loc or "
                "--use-prompt-refine in S2."
            )

        # Swin-B:
        # x_c1 channels = embed_dim;
        # x_c2 channels = 2 * embed_dim.
        model.small_refinement_head = (
            HighResolutionSmallObjectRefinementHead(
                x_c1_channels=embed_dim,
                x_c2_channels=2 * embed_dim,
                project_channels=(
                    args.small_project_channels
                ),
                hidden_channels=(
                    args.small_refine_channels
                ),
            )
        )

        print(
            "S2 high-resolution small-object refinement "
            "enabled: x_c1={}, x_c2={}, project={}, "
            "hidden={}".format(
                embed_dim,
                2 * embed_dim,
                args.small_project_channels,
                args.small_refine_channels,
            )
        )

    return model


def _load_model_lavt_one(pretrained, args):
    model = _segm_lavt_one(pretrained, args)
    return model


def lavt_one(pretrained='', args=None):
    return _load_model_lavt_one(pretrained, args)
