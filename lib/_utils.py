import torch
from torch import nn
from torch.nn import functional as F
from bert.modeling_bert import BertModel


def load_weights(model, load_path):
    dict_trained = torch.load(load_path)['model']
    dict_new = model.state_dict().copy()
    for key in dict_new.keys():
        if key in dict_trained.keys():
            dict_new[key] = dict_trained[key]
    model.load_state_dict(dict_new)
    del dict_new
    del dict_trained
    torch.cuda.empty_cache()
    print('load weights from {}'.format(load_path))
    return model


class _LAVTSimpleDecode(nn.Module):
    def __init__(self, backbone, classifier):
        super(_LAVTSimpleDecode, self).__init__()
        self.backbone = backbone
        self.classifier = classifier

    def forward(self, x, l_feats, l_mask):
        input_shape = x.shape[-2:]
        features = self.backbone(x, l_feats, l_mask)
        x_c1, x_c2, x_c3, x_c4 = features

        x = self.classifier(x_c4, x_c3, x_c2, x_c1)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=True)

        return x


class LAVT(_LAVTSimpleDecode):
    pass


###############################################
# LAVT One: put BERT inside the overall model #
###############################################
class _LAVTOneSimpleDecode(nn.Module):
    def __init__(self, backbone, classifier, args):
        super(_LAVTOneSimpleDecode, self).__init__()
        self.backbone = backbone
        self.classifier = classifier

        # E1 optional auxiliary localization head.
        self.coarse_head = None

        # E2 optional prompt-guided residual refinement head.
        self.refinement_head = None

        # S2 optional high-resolution refinement head.
        self.small_refinement_head = None
        self.text_encoder = BertModel.from_pretrained(args.ck_bert)
        self.text_encoder.pooler = None

    def forward(self, x, text, l_mask, t_mask, p_mask):
        input_shape = x.shape[-2:]
        ### language inference ###
        l_feats = self.text_encoder(text, attention_mask=l_mask)[0]
        l_feats = l_feats.permute(0, 2, 1)  # (B, 768, N_l)
        l_mask = l_mask.unsqueeze(dim=-1)  # (batch, N_l, 1)

        t_feats = self.text_encoder(text, attention_mask=t_mask)[0]
        t_feats = t_feats.permute(0, 2, 1)  # (B, 768, N_l)
        t_mask = t_mask.unsqueeze(dim=-1)  # (batch, N_l, 1)

        p_feats = self.text_encoder(text, attention_mask=p_mask)[0]
        p_feats = p_feats.permute(0, 2, 1)  # (B, 768, N_l)
        p_mask = p_mask.unsqueeze(dim=-1)  # (batch, N_l, 1)

        ##########################
        features = self.backbone(
            x,
            l_feats,
            l_mask,
            t_feats,
            t_mask,
            p_feats,
            p_mask,
        )

        # Deepest multimodal feature for coarse localization.
        x_c1, x_c2, x_c3, x_c4 = features

        coarse_logits = None
        if self.coarse_head is not None:
            coarse_logits = self.coarse_head(x_c4)

        # Original FIANet final decoder remains unchanged.
        final_logits = self.classifier(
            x_c4,
            x_c3,
            x_c2,
            x_c1,
        )

        final_logits = F.interpolate(
            final_logits,
            size=input_shape,
            mode='bilinear',
            align_corners=True,
        )

        # E2: use the coarse localization map as an internal
        # dense prompt to predict a residual correction.
        if self.refinement_head is not None:
            if coarse_logits is None:
                raise RuntimeError(
                    "E2 refinement requires coarse_logits"
                )

            residual_logits = self.refinement_head(
                x_c2,
                coarse_logits,
            )

            residual_logits = F.interpolate(
                residual_logits,
                size=input_shape,
                mode='bilinear',
                align_corners=False,
            )

            final_logits = final_logits + residual_logits

        # S2: restore small-object spatial details from
        # the two highest-resolution backbone features.
        if self.small_refinement_head is not None:
            if getattr(
                self.small_refinement_head,
                'requires_semantic_feature',
                False,
            ):
                small_residual_logits = (
                    self.small_refinement_head(
                        x_c1,
                        x_c2,
                        x_c4,
                    )
                )
            elif getattr(
                self.small_refinement_head,
                'requires_main_logits',
                False,
            ):
                small_residual_logits = (
                    self.small_refinement_head(
                        x_c1,
                        x_c2,
                        final_logits,
                    )
                )
            else:
                small_residual_logits = (
                    self.small_refinement_head(
                        x_c1,
                        x_c2,
                    )
                )

            small_residual_logits = F.interpolate(
                small_residual_logits,
                size=input_shape,
                mode='bilinear',
                align_corners=True,
            )

            final_logits = (
                final_logits
                + small_residual_logits
            )

        # E1/E2 need coarse output for auxiliary supervision.
        # Validation and test still receive final logits only.
        if self.training and coarse_logits is not None:
            return final_logits, coarse_logits

        return final_logits


class LAVTOne(_LAVTOneSimpleDecode):  #change
    pass
