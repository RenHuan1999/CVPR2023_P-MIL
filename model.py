import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

torch.set_default_tensor_type('torch.cuda.FloatTensor')


class Backbone_Proposal(torch.nn.Module):
    """
    Backbone for single modal in P-MIL framework
    """
    def __init__(self, feat_dim, n_class, dropout_ratio, roi_size):
        super().__init__()
        embed_dim = feat_dim // 2
        self.roi_size = roi_size

        self.prop_fusion = nn.Sequential(
            nn.Linear(feat_dim * 3, feat_dim),
            nn.ReLU(),
            nn.Dropout(dropout_ratio),
        )
        self.prop_classifier = nn.Sequential(
            nn.Conv1d(feat_dim, embed_dim, 1),
            nn.ReLU(),
            nn.Conv1d(embed_dim, n_class+1, 1),
        )
        self.prop_attention = nn.Sequential(
            nn.Conv1d(feat_dim, embed_dim, 1),
            nn.ReLU(),
            nn.Conv1d(embed_dim, 1, 1),
        )
        self.prop_completeness = nn.Sequential(
            nn.Conv1d(feat_dim, embed_dim, 1),
            nn.ReLU(),
            nn.Conv1d(embed_dim, 1, 1),
        )

    def forward(self, feat):
        """
        Inputs:
            feat: tensor of size [B, M, roi_size, D]

        Outputs:
            prop_cas:  tensor of size [B, C, M]
            prop_attn: tensor of size [B, 1, M]
            prop_iou:  tensor of size [B, 1, M]
        """
        feat1 = feat[:, :,                   : self.roi_size//6  , :].max(2)[0]
        feat2 = feat[:, :, self.roi_size//6  : self.roi_size//6*5, :].max(2)[0]
        feat3 = feat[:, :, self.roi_size//6*5:                   , :].max(2)[0]
        feat = torch.cat((feat2-feat1, feat2, feat2-feat3), dim=2)

        feat_fuse = self.prop_fusion(feat)                              # [B, M, D]
        feat_fuse = feat_fuse.transpose(-1, -2)                         # [B, D, M]

        prop_cas = self.prop_classifier(feat_fuse)                      # [B, C, M]
        prop_attn = self.prop_attention(feat_fuse)                      # [B, 1, M]
        prop_iou = self.prop_completeness(feat_fuse)                    # [B, 1, M]

        return prop_cas, prop_attn, prop_iou


class P_MIL(torch.nn.Module):
    """
    PyTorch module for the Proposal-based Multiple Instance Learning (P-MIL) framework
    """
    def __init__(self, args):
        super().__init__()
        n_class = args.num_class
        dropout_ratio = args.dropout_ratio
        self.feat_dim = args.feature_size
        self.max_proposal = args.max_proposal
        self.roi_size = args.roi_size

        self.prop_v_backbone = Backbone_Proposal(self.feat_dim // 2, n_class, dropout_ratio, self.roi_size)
        self.prop_f_backbone = Backbone_Proposal(self.feat_dim // 2, n_class, dropout_ratio, self.roi_size)

    def extract_roi_features(self, features, proposals, is_training):
        """
        Extract region of interest (RoI) features from raw i3d features based on given proposals

        Inputs:
            features: list of [T, D] tensors
            proposals: list of [M, 2] tensors
            is_training: bool

        Outputs:
            prop_features:tensor of size [B, M, roi_size, D]
            prop_mask: tensor of size [B, M]
        """
        num_prop = torch.tensor([prop.shape[0] for prop in proposals])
        batch, max_num = len(proposals), num_prop.max()
        # Limit the max number of proposals during training
        if is_training:
            max_num = min(max_num, self.max_proposal)
        prop_features = torch.zeros((batch, max_num, self.roi_size, self.feat_dim)).to(features[0].device)
        prop_mask = torch.zeros((batch, max_num)).to(features[0].device)

        for i in range(batch):
            feature = features[i]
            proposal = proposals[i]
            if num_prop[i] > max_num:
                sampled_idx = torch.randperm(num_prop[i])[:max_num]
                proposal = proposal[sampled_idx]

            # Extend the proposal by 25% of its length at both sides
            start, end = proposal[:, 0], proposal[:, 1]
            len_prop = end - start
            start_ext = start - 0.25 * len_prop
            end_ext = end + 0.25 * len_prop
            # Fill in blank at edge of the feature, offset 0.5, for more accurate RoI_Align results
            fill_len = torch.ceil(0.25 * len_prop.max()).long() + 1                         # +1 because of offset 0.5
            fill_blank = torch.zeros(fill_len, self.feat_dim).to(feature.device)
            feature = torch.cat([fill_blank, feature, fill_blank], dim=0)
            start_ext = start_ext + fill_len - 0.5
            end_ext = end_ext + fill_len - 0.5
            proposal_ext = torch.stack((start_ext, end_ext), dim=1)
            
            # Extract RoI features using RoI Align operation
            y1, y2 = proposal_ext[:, 0], proposal_ext[:, 1]
            x1, x2 = torch.zeros_like(y1), torch.ones_like(y2)
            boxes = torch.stack((x1, y1, x2, y2), dim=1)                                    # [M, 4]
            feature = feature.transpose(0, 1).unsqueeze(0).unsqueeze(3)                     # [1, D, T, 1]
            feat_roi = torchvision.ops.roi_align(feature, [boxes], [self.roi_size, 1])      # [M, D, roi_size, 1]
            feat_roi = feat_roi.squeeze(3).transpose(1, 2)                                  # [M, roi_size, D]
            prop_features[i, :proposal.shape[0], :, :] = feat_roi                           # [B, M, roi_size, D]
            prop_mask[i, :proposal.shape[0]] = 1                                            # [B, M]

        return prop_features, prop_mask

    def forward(self, features, proposals, is_training=True):
        """
        Inputs:
            features: list of [T, D] tensors
            proposals: list of [M, 2] tensors
            is_training: bool

        Outputs:
            outputs: dictionary
        """
        prop_features, prop_mask = self.extract_roi_features(features, proposals, is_training)
        prop_v_features = prop_features[..., :self.feat_dim // 2]
        prop_f_features = prop_features[..., self.feat_dim // 2:]

        prop_v_cas, prop_v_attn, prop_v_iou = self.prop_v_backbone(prop_v_features)
        prop_f_cas, prop_f_attn, prop_f_iou = self.prop_f_backbone(prop_f_features)

        outputs = {
            'prop_v_cas': prop_v_cas.transpose(-1, -2),     # [B, M, C]
            'prop_f_cas': prop_f_cas.transpose(-1, -2),     # [B, M, C]
            'prop_v_attn': prop_v_attn.transpose(-1, -2),   # [B, M, 1]
            'prop_f_attn': prop_f_attn.transpose(-1, -2),   # [B, M, 1]
            'prop_v_iou': prop_v_iou.transpose(-1, -2),     # [B, M, 1]
            'prop_f_iou': prop_f_iou.transpose(-1, -2),     # [B, M, 1]
            'prop_mask': prop_mask,                         # [B, M]
        }
        return outputs

    def get_consistency_weight(self, current, rampup_length):
        """
        Exponential rampup from https://arxiv.org/abs/1610.02242
        """
        if rampup_length == 0:
            return 1.0
        else:
            current = np.clip(current, 0.0, rampup_length)
            phase = 1.0 - current / rampup_length
            return float(np.exp(-5.0 * phase * phase))

    def segments_iou(self, segments1, segments2):
        """
        Inputs:
            segments1: tensor of size [M1, 2]
            segments2: tensor of size [M2, 2]

        Outputs:
            iou_temp: tensor of size [M1, M2]
        """
        segments1 = segments1.unsqueeze(1)                          # [M1, 1, 2]
        segments2 = segments2.unsqueeze(0)                          # [1, M2, 2]
        tt1 = torch.maximum(segments1[..., 0], segments2[..., 0])   # [M1, M2]
        tt2 = torch.minimum(segments1[..., 1], segments2[..., 1])   # [M1, M2]
        intersection = tt2 - tt1
        union = (segments1[..., 1] - segments1[..., 0]) + (segments2[..., 1] - segments2[..., 0]) - intersection
        iou = intersection / (union + 1e-6)                         # [M1, M2]
        # Remove negative values
        iou_temp = torch.zeros_like(iou)
        iou_temp[iou > 0] = iou[iou > 0]
        return iou_temp

    def criterion(self, outputs, labels, proposals, epoch, args):
        """
        Compute the total loss function

        Inputs: 
            outputs: dictionary
            labels: tensor of size [B, C]
            proposals: list of [M, 2] tensors
            epoch: int
            args: argparse.Namespace

        Outputs:
            loss_dict: dictionary
        """
        prop_v_cas, prop_v_attn, prop_v_iou = outputs['prop_v_cas'], outputs['prop_v_attn'], outputs['prop_v_iou']
        prop_f_cas, prop_f_attn, prop_f_iou = outputs['prop_f_cas'], outputs['prop_f_attn'], outputs['prop_f_iou']
        prop_mask = outputs['prop_mask']

        prop_v_attn = torch.sigmoid(prop_v_attn)                        # [B, M, 1]
        prop_f_attn = torch.sigmoid(prop_f_attn)                        # [B, M, 1]
        prop_v_iou = torch.sigmoid(prop_v_iou)                          # [B, M, 1]
        prop_f_iou = torch.sigmoid(prop_f_iou)                          # [B, M, 1]
        prop_mask = prop_mask.unsqueeze(2).bool()                       # [B, M, 1]
        prop_mask_cas = prop_mask.repeat((1, 1, prop_v_cas.shape[2]))   # [B, M, C]

        # proposal classification loss
        prop_v_cas_supp = prop_v_cas * prop_v_attn
        prop_f_cas_supp = prop_f_cas * prop_f_attn
        loss_prop_mil_orig_v = self.prop_topk_loss(prop_v_cas,      labels, prop_mask_cas, is_back=True,  topk=args.k)
        loss_prop_mil_orig_f = self.prop_topk_loss(prop_f_cas,      labels, prop_mask_cas, is_back=True,  topk=args.k)
        loss_prop_mil_supp_v = self.prop_topk_loss(prop_v_cas_supp, labels, prop_mask_cas, is_back=False, topk=args.k)
        loss_prop_mil_supp_f = self.prop_topk_loss(prop_f_cas_supp, labels, prop_mask_cas, is_back=False, topk=args.k)

        # Instance-level Rank Consistency (IRC) loss
        loss_prop_irc_v = self.prop_irc_loss(prop_v_cas, prop_f_cas, prop_f_attn, labels, prop_mask, prop_mask_cas, proposals)
        loss_prop_irc_f = self.prop_irc_loss(prop_f_cas, prop_v_cas, prop_v_attn, labels, prop_mask, prop_mask_cas, proposals)

        # proposal completeness loss
        loss_prop_comp_v = self.prop_comp_loss(prop_v_iou, prop_f_attn, prop_mask, proposals, args.gamma)
        loss_prop_comp_f = self.prop_comp_loss(prop_f_iou, prop_v_attn, prop_mask, proposals, args.gamma)

        loss_prop_mil_orig = args.weight_loss_prop_mil_orig * (loss_prop_mil_orig_v + loss_prop_mil_orig_f) / 2
        loss_prop_mil_supp = args.weight_loss_prop_mil_supp * (loss_prop_mil_supp_v + loss_prop_mil_supp_f) / 2
        loss_prop_irc = args.weight_loss_prop_irc * (loss_prop_irc_v + loss_prop_irc_f) / 2 * self.get_consistency_weight(epoch, args.rampup_length)
        loss_prop_comp = args.weight_loss_prop_comp * (loss_prop_comp_v + loss_prop_comp_f) / 2 * self.get_consistency_weight(epoch, args.rampup_length)
        loss_total = loss_prop_mil_orig + loss_prop_mil_supp + loss_prop_irc + loss_prop_comp

        loss_dict = {
            'loss_total': loss_total,
            'loss_prop_mil_orig': loss_prop_mil_orig,
            'loss_prop_mil_supp': loss_prop_mil_supp,
            'loss_prop_irc': loss_prop_irc,
            'loss_prop_comp': loss_prop_comp,
        }
        return loss_dict

    def prop_topk_loss(self, cas, labels, mask_cas, is_back=True, topk=8):
        """
        Compute the topk classification loss

        Inputs:
            cas: tensor of size [B, M, C]
            labels: tensor of size [B, C]
            mask_cas: tensor of size [B, M, C]
            is_back: bool
            topk: int

        Outputs:
            loss_mil: tensor
        """
        if is_back:
            labels_with_back = torch.cat((labels, torch.ones_like(labels[:, [0]])), dim=-1)
        else:
            labels_with_back = torch.cat((labels, torch.zeros_like(labels[:, [0]])), dim=-1)
        labels_with_back = labels_with_back / (torch.sum(labels_with_back, dim=-1, keepdim=True) + 1e-4)

        loss_mil = 0
        for b in range(cas.shape[0]):
            cas_b = cas[b][mask_cas[b]].reshape((-1, cas.shape[-1]))
            topk_val, _ = torch.topk(cas_b, k=max(1, int(cas_b.shape[-2] // topk)), dim=-2)
            video_score = torch.mean(topk_val, dim=-2)
            loss_mil += - (labels_with_back[b] * F.log_softmax(video_score, dim=-1)).sum(dim=-1).mean()
        loss_mil /= cas.shape[0]

        return loss_mil

    def prop_irc_loss(self, cas_stu, cas_tea, attn, labels, mask, mask_cas, proposals):
        """
        Compute the Instance-level Rank Consistency (IRC) loss

        Inputs:
            cas_stu: tensor of size [B, M, C]
            cas_tea: tensor of size [B, M, C]
            attn: tensor of size [B, M, 1]
            labels: tensor of size [B, C]
            mask: bool tensor of size [B, M, 1]
            mask_cas: bool tensor of size [B, M, C]
            proposals: list of [M, 2] tensors

        Outputs:
            loss_irc: tensor
        """
        loss_irc = 0
        for b in range(len(proposals)):
            attn_b = attn[b][mask[b]]
            cas_stu_b = cas_stu[b][mask_cas[b]].reshape((-1, mask_cas.shape[-1]))
            cas_tea_b = cas_tea[b][mask_cas[b]].reshape((-1, mask_cas.shape[-1]))
            proposals_iou = self.segments_iou(proposals[b], proposals[b])
            # used to mask out non-overlapping proposals
            proposals_mask = torch.zeros_like(proposals_iou)
            proposals_mask[proposals_iou <= 0] = -1e3
            proposals_mask[proposals_iou > 0] = 0

            loss_irc_b = 0
            for c in torch.where(labels[b])[0]:
                score_stu = cas_stu_b[:, c]
                score_tea = cas_tea_b[:, c]

                # the KL loss is only computed for proposals that overlap with the given proposal
                softmax_tea = F.softmax(proposals_mask + score_tea.unsqueeze(0), dim=1)
                softmax_stu = F.log_softmax(proposals_mask + score_stu.unsqueeze(0), dim=1)
                loss_kl_matrix = F.kl_div(softmax_stu, softmax_tea.detach(), reduction='none').sum(-1)

                # eliminate the low-confidence proposals
                retained = attn_b > torch.mean(attn_b)
                loss_irc_b += loss_kl_matrix[retained].mean()
            loss_irc_b /= labels[b].sum()
            loss_irc += loss_irc_b
        loss_irc /= len(proposals)

        return loss_irc

    def prop_comp_loss(self, pred_iou, attn, mask, proposals, gamma):
        """
        Compute the completeness loss

        Inputs:
            pred_iou: tensor of size [B, M, 1]
            attn: tensor of size [B, M, 1]
            mask: bool tensor of size [B, M, 1]
            proposals: list of [M, 2] tensors
            gamma: float

        Outputs:
            loss_comp: tensor
        """
        loss_comp = 0
        for b in range(len(proposals)):
            attn_b = attn[b][mask[b]]
            pred_iou_b = pred_iou[b][mask[b]]
            proposals_iou = self.segments_iou(proposals[b], proposals[b])
            proposals_mask = proposals_iou > 0

            # using NMS to select the pseudo instances, the running speed is slow
            choiced = []
            retained = attn_b > gamma * torch.max(attn_b)
            while retained.sum() > 0:
                max_idx = torch.max(attn_b[retained], dim=0)[1]
                max_idx = torch.where(retained)[0][max_idx]
                overlap = proposals_mask[max_idx]
                retained[overlap] = False
                choiced.append(max_idx)
            choiced = torch.stack(choiced, dim=0)
            pseudo_instances = proposals[b][choiced]

            pseudo_iou = self.segments_iou(proposals[b], pseudo_instances)
            pseudo_iou = torch.max(pseudo_iou, dim=1)[0]
            loss_comp += F.mse_loss(pred_iou_b, pseudo_iou)
        loss_comp /= len(proposals)

        return loss_comp

