import types
import torch
import transformers
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import numpy as np
import pdb
import math
from .Tool_model import AutomaticWeightedLoss
from .UMAEA_tools import MultiModalEncoder, VirEmbGen
from .UMAEA_loss import CustomMultiLossLayer, icl_loss, ial_loss

from src.utils import pairwise_distances
import os.path as osp
import json


class UMAEA(nn.Module):
    def __init__(self, kgs, args):
        super().__init__()
        self.kgs = kgs
        self.args = args
        self.img_features = F.normalize(torch.FloatTensor(kgs["images_list"])).cuda()
        self.input_idx = kgs["input_idx"].cuda()
        self.adj = kgs["adj"].cuda()
        self.rel_features = torch.Tensor(kgs["rel_features"]).cuda()
        self.att_features = torch.Tensor(kgs["att_features"]).cuda()
        self.ent_wo_img = torch.tensor(kgs['ent_wo_img']).cuda()
        self.ent_w_img = torch.tensor(kgs['ent_w_img']).cuda()
        self.name_features = None
        self.char_features = None
        self.train_ill = kgs["train_ill"]
        self.args.modal_num = 3
        if kgs["name_features"] is not None:
            self.name_features = kgs["name_features"].cuda()
            self.char_features = kgs["char_features"].cuda()
            self.args.modal_num = 4
        img_dim = self._get_img_dim(kgs)
        char_dim = kgs["char_features"].shape[1] if self.char_features is not None else 100

        self.multimodal_encoder = MultiModalEncoder(args=self.args,
                                                    ent_num=kgs["ent_num"],
                                                    img_feature_dim=img_dim,
                                                    char_feature_dim=char_dim,
                                                    use_project_head=self.args.use_project_head,
                                                    attr_input_dim=kgs["att_features"].shape[1])

        self.multi_loss_layer = CustomMultiLossLayer(loss_num=6)  # 6
        self.multi_loss_layer_2 = CustomMultiLossLayer(loss_num=7)
        self.multi_loss_layer_3 = AutomaticWeightedLoss(num=7)  # 5
        self.criterion_cl = icl_loss(tau=self.args.tau, ab_weight=self.args.ab_weight, n_view=2)
        self.criterion_cl_joint = icl_loss(tau=self.args.tau, ab_weight=self.args.ab_weight, n_view=2, neg_cross_kg=self.args.neg_cross_kg)

        self.criterion_align = ial_loss(tau=self.args.tau2, ab_weight=self.args.ab_weight, zoom=self.args.zoom, reduction=self.args.reduction)
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")
        self.criterion = nn.MSELoss()

        tmp = -1 * torch.ones(self.input_idx.shape[0], dtype=torch.int64).cuda()
        self.idx_one = torch.ones(self.args.batch_size, dtype=torch.int64).cuda()
        self.idx_double = torch.cat([self.idx_one, self.idx_one]).cuda()
        self.last_num = 1000000000000
        # self.idx_one = np.ones(self.args.batch_size, dtype=np.int64)

    def forward(self, batch):
        gph_emb, img_emb, rel_emb, att_emb, name_emb, char_emb, joint_emb, joint_emb_fz, hidden_states, weight_norm, vir_emb, x_hat, hyb_emb, kl_div = self.joint_emb_generat(only_joint=False)
        gph_emb_hid, rel_emb_hid, att_emb_hid, img_emb_hid, name_emb_hid, char_emb_hid, joint_emb_hid = self.generate_hidden_emb(hidden_states)

        if self.args.stage == 1:
            batch_tensor = torch.from_numpy(batch).view(-1).type(torch.long).cuda()
            intersection = torch.tensor(list(set(batch_tensor.tolist()) & set(self.ent_w_img.tolist()))).cuda()

            target_img_emb = img_emb[intersection, :]
            source_img_emb = vir_emb[intersection, :]
            if self.args.adapt == "vae":
                target_hyb_emb = hyb_emb[intersection, :]
                source_hyb_emb = x_hat[intersection, :]
                # L_{Re}^{hyb}
                loss_vir_hyb = 6000 * self.kl_loss(F.log_softmax(source_hyb_emb, dim=1), F.softmax(target_hyb_emb, dim=1))
                # L_{Sim}
                loss_kl = 6000 * self.criterion_align(vir_emb, hyb_emb, self.train_ill)
            # L_{Re}^{vis}
            loss_vir = 6000 * self.kl_loss(F.log_softmax(source_img_emb, dim=1), F.softmax(target_img_emb, dim=1))

        # Global Modality Integration
        GMI_loss = self.criterion_cl_joint(joint_emb, batch) + self.criterion_cl_joint(joint_emb_fz, batch)

        # Entity-level Modality Alignment
        ECIA_loss = self.inner_view_loss(gph_emb, rel_emb, att_emb, img_emb, name_emb, char_emb, batch, vir_emb=vir_emb, weight_norm=weight_norm)

        # Late Modality Refinement
        IIR_loss = self.inner_view_loss(gph_emb_hid, rel_emb_hid, att_emb_hid, img_emb_hid, name_emb_hid, char_emb_hid, batch)

        loss_list = [GMI_loss, ECIA_loss, IIR_loss]
        loss_update = {}
        if self.args.stage == 1:
            loss_list.append(loss_vir)
            loss_update = {"loss_vir": loss_vir.item()}
            if self.args.adapt == "vae":
                # add L_{KL}
                loss_list.extend([loss_vir_hyb, 3000 * kl_div, loss_kl])
                loss_update = {"loss_vir": loss_vir.item(), "loss_vir_hyb": loss_vir_hyb.item(), "kl_div": kl_div.item(), "kl_loss": loss_kl.item()}

        if self.args.awloss:
            loss_all = self.multi_loss_layer_3(loss_list)
        else:
            loss_all = sum(loss_list)

        loss_dic = {"joint_Intra_modal": GMI_loss.item(), "Intra_modal": ECIA_loss.item(), "IIR_loss": IIR_loss.item()}
        loss_dic.update(loss_update)
        output = {"loss_dic": loss_dic, "emb": joint_emb}
        return loss_all, output

    def generate_hidden_emb(self, hidden):
        gph_emb = F.normalize(hidden[:, 0, :].squeeze(1))
        rel_emb = F.normalize(hidden[:, 1, :].squeeze(1))
        att_emb = F.normalize(hidden[:, 2, :].squeeze(1))
        if self.args.w_img:
            img_emb = F.normalize(hidden[:, 3, :].squeeze(1))
        else:
            img_emb = None
        if hidden.shape[1] >= 6:
            name_emb = F.normalize(hidden[:, 4, :].squeeze(1))
            char_emb = F.normalize(hidden[:, 5, :].squeeze(1))
        else:
            name_emb, char_emb = None, None
            loss_name, loss_char = None, None
        emb_list = [gph_emb, rel_emb, att_emb, img_emb, name_emb, char_emb]
        emb_cat = [i for i in emb_list if i is not None]
        joint_emb = torch.cat(emb_cat, dim=1)

        return gph_emb, rel_emb, att_emb, img_emb, name_emb, char_emb, joint_emb

    def inner_view_loss(self, gph_emb, rel_emb, att_emb, img_emb, name_emb, char_emb, train_ill, vir_emb=None, weight_norm=None):
        if weight_norm is not None:
            mod_num = weight_norm.shape[1]
            weight_norm = weight_norm * mod_num
            loss_GCN = self.criterion_cl(gph_emb, train_ill, weight_norm=weight_norm[:, 3]) if gph_emb is not None else 0
            loss_rel = self.criterion_cl(rel_emb, train_ill, weight_norm=weight_norm[:, 2]) if rel_emb is not None else 0
            loss_att = self.criterion_cl(att_emb, train_ill, weight_norm=weight_norm[:, 1]) if att_emb is not None else 0
            loss_img = self.criterion_cl(img_emb, train_ill, weight_norm=weight_norm[:, 0]) if img_emb is not None else 0
            loss_vir = self.criterion_cl(vir_emb, train_ill, weight_norm=weight_norm[:, 0]) if vir_emb is not None else 0
            loss_name = self.criterion_cl(name_emb, train_ill, weight_norm=weight_norm[:, 4]) if name_emb is not None else 0
            loss_char = self.criterion_cl(char_emb, train_ill, weight_norm=weight_norm[:, 5]) if char_emb is not None else 0
        else:
            loss_GCN = self.criterion_cl(gph_emb, train_ill) if gph_emb is not None else 0
            loss_rel = self.criterion_cl(rel_emb, train_ill) if rel_emb is not None else 0
            loss_att = self.criterion_cl(att_emb, train_ill) if att_emb is not None else 0
            loss_img = self.criterion_cl(img_emb, train_ill) if img_emb is not None else 0
            loss_vir = self.criterion_cl(vir_emb, train_ill) if vir_emb is not None else 0
            loss_name = self.criterion_cl(name_emb, train_ill) if name_emb is not None else 0
            loss_char = self.criterion_cl(char_emb, train_ill) if char_emb is not None else 0
        total_loss = self.multi_loss_layer([loss_GCN, loss_rel, loss_att, loss_img, loss_name, loss_char]) + loss_vir

        return total_loss

    # --------- necessary ---------------

    def joint_emb_generat(self, only_joint=True, test=False):
        gph_emb, img_emb, rel_emb, att_emb, \
            name_emb, char_emb, joint_emb, joint_emb_fz, hidden_states, weight_norm, vir_emb, x_hat, hyb_emb, kl_div = self.multimodal_encoder(self.input_idx,
                                                                                                                                               self.adj,
                                                                                                                                               self.img_features,
                                                                                                                                               self.rel_features,
                                                                                                                                               self.att_features,
                                                                                                                                               self.name_features,
                                                                                                                                               self.char_features,
                                                                                                                                               ent_wo_img=self.ent_wo_img,
                                                                                                                                               _test=test)

        if only_joint:
            return joint_emb_fz, weight_norm
        else:
            return gph_emb, img_emb, rel_emb, att_emb, name_emb, char_emb, joint_emb, joint_emb_fz, hidden_states, weight_norm, vir_emb, x_hat, hyb_emb, kl_div

    # --------- share ---------------

    def _get_img_dim(self, kgs):
        if isinstance(kgs["images_list"], list):
            img_dim = kgs["images_list"][0].shape[1]
        elif isinstance(kgs["images_list"], np.ndarray) or torch.is_tensor(kgs["images_list"]):
            img_dim = kgs["images_list"].shape[1]
        return img_dim

    def Iter_new_links(self, epoch, left_non_train, final_emb, right_non_train, new_links=[]):
        if len(left_non_train) == 0 or len(right_non_train) == 0:
            return new_links
        distance_list = []
        for i in np.arange(0, len(left_non_train), 1000):
            d = pairwise_distances(final_emb[left_non_train[i:i + 1000]], final_emb[right_non_train])
            distance_list.append(d)
        distance = torch.cat(distance_list, dim=0)
        preds_l = torch.argmin(distance, dim=1).cpu().numpy().tolist()
        preds_r = torch.argmin(distance.t(), dim=1).cpu().numpy().tolist()
        del distance_list, distance, final_emb
        if (epoch + 1) % (self.args.semi_learn_step * 5) == self.args.semi_learn_step:
            new_links = [(left_non_train[i], right_non_train[p]) for i, p in enumerate(preds_l) if preds_r[p] == i]
        else:
            new_links = [(left_non_train[i], right_non_train[p]) for i, p in enumerate(preds_l) if (preds_r[p] == i) and ((left_non_train[i], right_non_train[p]) in new_links)]

        return new_links

    def data_refresh(self, logger, train_ill, test_ill_, left_non_train, right_non_train, new_links=[]):
        if len(new_links) != 0 and (len(left_non_train) != 0 and len(right_non_train) != 0):
            new_links_select = new_links
            train_ill = np.vstack((train_ill, np.array(new_links_select)))
            num_true = len([nl for nl in new_links_select if nl in test_ill_])
            for nl in new_links_select:
                left_non_train.remove(nl[0])
                right_non_train.remove(nl[1])

            logger.info(f"#new_links_select:{len(new_links_select)}")
            logger.info(f"train_ill.shape:{train_ill.shape}")
            logger.info(f"#true_links: {num_true}")
            logger.info(f"true link ratio: {(100 * num_true / len(new_links_select)):.1f}%")
            logger.info(f"#entity not in train set: {len(left_non_train)} (left) {len(right_non_train)} (right)")

            new_links = []
        else:
            logger.info("len(new_links) is 0")

        return left_non_train, right_non_train, train_ill, new_links
