import os
import os.path as osp
import torch
import torch.nn as nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, RandomSampler
from torch.cuda.amp import GradScaler, autocast
from datetime import datetime
from easydict import EasyDict as edict
from tqdm import tqdm
import pdb
import pprint
import json
import pickle
from collections import defaultdict

from config import cfg
from torchlight import initialize_exp, set_seed, get_dump_path
from src.data import load_data, Collator_base, EADataset
from src.utils import set_optim, Loss_log, pairwise_distances, csls_sim
from model import UMAEA
from model.Tool_model import AutomaticWeightedLoss, MAS, RWalk

import torch.nn.functional as F

import scipy
import gc
import copy


class Runner:
    def __init__(self, args, writer=None, logger=None):
        self.datapath = edict()
        self.datapath.log_dir = get_dump_path(args)
        self.datapath.model_dir = os.path.join(self.datapath.log_dir, 'model')
        self.args = args
        self.writer = writer
        self.logger = logger
        self.scaler = GradScaler()
        self.model_list = []
        self.epoch = -1
        set_seed(args.random_seed)
        self.data_init()
        self.model_choise()

        self.model.cuda()
        set_seed(args.random_seed)

        self.stage_epoch = [int(x) for x in self.args.stage_epoch.strip().split(",")]
        if self.args.stage_epoch == "":
            self.il_stage_epoch = self.stage_epoch
        else:
            self.il_stage_epoch = [int(x) for x in self.args.il_stage_epoch.strip().split(",")]
        self.stage_lr = [5e-4, 4e-3, 2e-4]
        self.stage_wd = [1e-4, 3, 1e-4]
        self.stage = self.args.stage

        if self.args.only_test:
            self.dataloader_init(test_set=self.test_set)
        else:
            self.dataloader_init(train_set=self.train_set, eval_set=self.eval_set, test_set=self.test_set)

            self.model_list = [self.model]

            self.args.epoch = sum(self.stage_epoch[self.stage:])
            self.next_stage_epoch = self.stage_epoch[self.stage]
            self.args.lr = self.stage_lr[self.stage]
            self.args.weight_decay = self.stage_wd[self.stage]
            if self.args.il:
                self.args.il_start = self.args.epoch
                self.args.epoch += sum(self.il_stage_epoch)
            self.optim_init(self.args, total_epoch=self.next_stage_epoch)

    def model_choise(self, load_name=None, plug=False):
        assert self.args.model_name in ["EVA", "MCLEA", "MSNEA", "UMAEA"]
        if self.args.model_name == "UMAEA":
            self.model = UMAEA(self.KGs, self.args)

        first_time_load = False
        if load_name is None:
            load_name = self.args.model_name_save
            first_time_load = True
        self.model = self._load_model(self.model, model_name=load_name, first_time_load=first_time_load)
        self.model.multi_loss_layer_3.params.data = torch.ones(7, requires_grad=True).cuda()

        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info(f"total params num: {total_params}")

    def optim_init(self, opt, total_step=None, total_epoch=None, accumulation_step=None):
        step_per_epoch = len(self.train_dataloader)
        if total_epoch is not None:
            opt.total_steps = int(step_per_epoch * total_epoch)
        else:
            opt.total_steps = int(step_per_epoch * opt.epoch) if total_step is None else int(total_step)
        opt.warmup_steps = int(opt.total_steps * 0.15)

        if total_step is None:
            self.logger.info(f"warmup_steps: {opt.warmup_steps}")
            self.logger.info(f"total_steps: {opt.total_steps}")
            self.logger.info(f"weight_decay: {opt.weight_decay}")

        no_freeze_part = []
        freeze_part = []

        if self.stage == 1:
            no_freeze_part.extend(["vir_emb_gen", "multi_loss_layer_2"])
        if self.stage == 2:
            freeze_part.extend(["vir_emb_gen", "rel_fc", "entity_emb", "cross_graph_model", "att_fc"])
        self.optimizer, self.scheduler = set_optim(opt, [self.model], freeze_part, no_freeze_part, accumulation_step)

    def data_init(self):
        self.KGs, self.non_train, self.train_set, self.eval_set, self.test_set, self.test_ill_ = load_data(self.logger, self.args)
        self.train_ill = self.train_set.data
        self.eval_left = torch.LongTensor(self.eval_set[:, 0].squeeze()).cuda()
        self.eval_right = torch.LongTensor(self.eval_set[:, 1].squeeze()).cuda()
        if self.test_set is not None:
            self.test_left = torch.LongTensor(self.test_ill[:, 0].squeeze()).cuda()
            self.test_right = torch.LongTensor(self.test_ill[:, 1].squeeze()).cuda()

        self.eval_sampler = None

    def dataloader_init(self, train_set=None, eval_set=None, test_set=None):
        bs = self.args.batch_size
        collator = Collator_base(self.args)

        self.args.workers = min([os.cpu_count(), self.args.batch_size, self.args.workers])
        if train_set is not None:
            self.train_dataloader = self._dataloader(train_set, bs, collator)
        if test_set is not None:
            self.test_dataloader = self._dataloader(test_set, bs, collator)
        if eval_set is not None:
            self.eval_dataloader = self._dataloader(eval_set, bs, collator)

    def _dataloader(self, train_set, batch_size, collator):
        train_dataloader = DataLoader(
            train_set,
            num_workers=self.args.workers,
            persistent_workers=True,  # True
            shuffle=(self.args.only_test == 0),
            drop_last=False,
            batch_size=batch_size,
            collate_fn=collator
        )
        return train_dataloader

    def run(self):
        self.loss_log = Loss_log()
        self.curr_loss = 0.
        self.lr = self.args.lr
        self.curr_loss_dic = defaultdict(float)
        self.weight = [1, 1, 1, 1, 1, 1]
        self.loss_weight = [1, 1]
        self.loss_item = 99999.
        self.step = 1
        self.epoch = -1
        self.new_links = []
        self.best_model_wts = None
        self.best_mrr = 0
        self.early_stop_init = 300
        self.early_stop_count = self.early_stop_init
        self.il_stage = 0
        pfx_train = "Norm"
        with tqdm(total=self.args.epoch) as _tqdm:
            for i in range(self.args.epoch):
                # _tqdm.set_description(f'Train | epoch {i} Loss {self.loss_log.get_loss():.5f} Acc {self.loss_log.get_acc()*100:.3f}%')
                # -------------------------------
                self.epoch += 1
                torch.cuda.empty_cache()
                if self.epoch == self.next_stage_epoch:
                    if self.args.il and self.stage == 2 and self.il_stage == 0:
                        self.il_stage = 1
                        pfx_train = "IL"
                        self.stage_epoch = self.il_stage_epoch

                    if self.best_model_wts is not None:
                        self.logger.info("load from the best model before IL... ")
                        self.model.load_state_dict(self.best_model_wts)
                        self.best_model_wts = None

                    name = self._save_name_define(_pfx=f"_{pfx_train}")
                    if self.stage_epoch[self.stage] > 0:
                        self.test(save_name=f"{name}_test_ep{self.args.epoch}_stg_{self.stage}")
                    if not self.args.only_test:
                        self._save_model(self.model, input_name=name)
                    self.model_choise(load_name=name, plug=True)

                    self.stage = (self.stage + 1) % 3
                    if self.stage_epoch[self.stage] > 0:
                        self.model.args.stage = self.stage
                        self.model.multimodal_encoder.args = self.model.args
                        self.step = 1
                        self.next_stage_epoch = self.epoch + self.stage_epoch[self.stage]
                        self.args.lr = self.stage_lr[self.stage]
                        self.args.weight_decay = self.stage_wd[self.stage]
                        total_epoch = self.stage_epoch[self.stage]
                        self.logger.info(f"***********  Switch to stage [{self.stage}]-[{pfx_train}]: LR {self.args.lr} Epoch {total_epoch}  ***********")

                        if self.args.il and self.il_stage == 1:
                            self.args.lr /= 5
                            self.eval_epoch = 1
                            if self.stage == 0:
                                total_epoch *= 3
                    else:
                        self.next_stage_epoch += 1
                        continue

                    self.optim_init(self.args, total_epoch=total_epoch)

                if self.il_stage == 1 and (self.epoch + 1) % self.args.semi_learn_step == 0 and self.args.il and self.stage == 0:
                    self.il_for_ea()

                if self.il_stage == 1 and (self.epoch + 1) % (self.args.semi_learn_step * 10) == 0 and len(self.new_links) != 0 and self.args.il and self.stage == 0:
                    self.il_for_data_ref()
                    # pass

                self.train(_tqdm)
                self.loss_log.update(self.curr_loss)
                self.loss_item = self.loss_log.get_loss()
                _tqdm.set_description(
                    f'Train | Stg [{self.stage}] Ep [{self.epoch}/{self.args.epoch}] Step [{self.step}/{self.args.total_steps}] LR [{self.lr:.5f}] Loss {self.loss_log.get_loss():.5f} ')
                self.update_loss_log()
                if (i + 1) % self.args.eval_epoch == 0:
                    self.eval()
                _tqdm.update(1)
                if self.il_stage == 1 and self.early_stop_count <= 0:
                    logger.info(f"Early stop in epoch {self.epoch}")
                    break

        name = self._save_name_define()
        if self.best_model_wts is not None:
            self.logger.info("load from the best model before final testing ... ")
            self.model.load_state_dict(self.best_model_wts)
        self.test(save_name=f"{name}_test_ep{self.args.epoch}_seed{self.args.random_seed}")

        self.logger.info(f"min loss {self.loss_log.get_min_loss()}")
        if not self.args.only_test and self.args.save_model:
            self._save_model(self.model, input_name=name)

    def il_for_ea(self):
        with torch.no_grad():
            if self.args.model_name in ["UMAEA"]:
                final_emb, weight_norm = self.model.joint_emb_generat()
            else:
                final_emb = self.model.joint_emb_generat()
            final_emb = F.normalize(final_emb)
            self.new_links = self.model.Iter_new_links(self.epoch, self.non_train["left"], final_emb, self.non_train["right"], new_links=self.new_links)
            if (self.epoch + 1) % (self.args.semi_learn_step * 5) == 0:
                self.logger.info(f"[epoch {self.epoch}] #links in candidate set: {len(self.new_links)}")

    def il_for_data_ref(self):
        self.non_train["left"], self.non_train["right"], self.train_ill, self.new_links = self.model.data_refresh(
            self.logger, self.train_ill, self.test_ill_, self.non_train["left"], self.non_train["right"], new_links=self.new_links)
        set_seed(self.args.random_seed)
        self.train_set = EADataset(self.train_ill)
        self.dataloader_init(train_set=self.train_set)
        self.model.train_ill = self.train_ill

    def _save_name_define(self, _pfx=""):
        prefix = ""
        if self.args.stage != 0:
            prefix = f"Retrain_stg_{self.stage}"
        if self.args.il:
            prefix = f"il{self.args.epoch-self.args.il_start}_b{self.args.il_start}_{prefix}{_pfx}"
        name = f'{self.args.exp_id}_{prefix}_seed{self.args.random_seed}_R{self.args.ratio}'
        return name

    def train(self, _tqdm):
        self.model.train()
        curr_loss = 0.
        self.loss_log.acc_init()
        accumulation_steps = self.args.accumulation_steps
        for batch in self.train_dataloader:
            loss, output = self.model(batch)
            loss = loss / accumulation_steps
            self.scaler.scale(loss).backward()

            self.step += 1
            curr_loss += loss.item()
            self.output_statistic(loss, output)

            if self.step % accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)
                for model in self.model_list:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.clip)
                scale = self.scaler.get_scale()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                skip_lr_sched = (scale > self.scaler.get_scale())
                if not skip_lr_sched:
                    self.scheduler.step()
                self.lr = self.scheduler.get_last_lr()[-1]
                self.writer.add_scalars("lr", {"lr": self.lr}, self.step)
                for model in self.model_list:
                    model.zero_grad(set_to_none=True)

        return curr_loss

    def output_statistic(self, loss, output):
        self.curr_loss += loss.item()
        if output is None:
            return
        for key in output['loss_dic'].keys():
            self.curr_loss_dic[key] += output['loss_dic'][key]
        if 'weight' in output and output['weight'] is not None:
            self.weight = output['weight']
        if 'loss_weight' in output and output['loss_weight'] is not None:
            self.loss_weight = output['loss_weight']

    def update_loss_log(self):
        vis_dict = {"train_loss": self.curr_loss}
        vis_dict.update(self.curr_loss_dic)
        self.writer.add_scalars("loss", vis_dict, self.step)

        if self.weight is not None:
            weight_dic = {}
            weight_dic["img"] = self.weight[0]
            weight_dic["attr"] = self.weight[1]
            weight_dic["rel"] = self.weight[2]
            weight_dic["graph"] = self.weight[3]
            if self.args.w_name or self.args.w_char:
                weight_dic["name"] = self.weight[4]
                weight_dic["char"] = self.weight[5]
            self.writer.add_scalars("modal_weight", weight_dic, self.step)

        if self.loss_weight is not None and self.loss_weight != [1, 1]:
            weight_dic = {}
            weight_dic["mask"] = 1 / (self.loss_weight[0]**2)
            weight_dic["kpi"] = 1 / (self.loss_weight[1]**2)
            self.writer.add_scalars("loss_weight", weight_dic, self.step)

        self.curr_loss = 0.
        for key in self.curr_loss_dic:
            self.curr_loss_dic[key] = 0.

    def eval(self, last_epoch=False, save_name=""):
        test_left = self.eval_left
        test_right = self.eval_right
        self.model.eval()
        self._test(test_left, test_right, last_epoch=last_epoch, save_name=save_name, test=True)

    def test(self, save_name="", last_epoch=True):
        if self.test_set is None:
            test_left = self.eval_left
            test_right = self.eval_right
        else:
            test_left = self.test_left
            test_right = self.test_right
        self.model.eval()
        self.logger.info(" --------------------- Test result Norm--------------------- ")
        self._test(test_left, test_right, last_epoch=last_epoch, save_name=save_name, test=True)
        self.logger.info(" --------------------- Test result --------------------- ")
        self._test(test_left, test_right, last_epoch=last_epoch, save_name=save_name)

    def _test(self, test_left, test_right, last_epoch=False, save_name="", loss=None, test=False):
        with torch.no_grad():
            w_normalized = None
            if self.args.model_name in ["EVA", "MCLEA"]:
                if self.args.model_name == "EVA":
                    self.model.emb_generat()
                    if self.args.w_name and self.args.w_char:
                        w_normalized = F.softmax(self.model.weight_raw, dim=0)
                    else:
                        w_normalized = F.softmax(self.model.weight_raw[:4], dim=0)
                else:
                    w_normalized = F.softmax(self.model.multimodal_encoder.fusion.weight.reshape(-1), dim=0)

                appdx = ""
                if self.args.w_name and self.args.w_char:
                    appdx = f"-[name_{w_normalized[4]:.3f}]-[char_{w_normalized[5]:.3f}]"
                self.logger.info(f"weight_raw:[img_{w_normalized[0]:.3f}]-[attr_{w_normalized[1]:.3f}]-[rel_{w_normalized[2]:.3f}]-[graph_{w_normalized[3]:.3f}]{appdx}")
            if self.args.model_name in ["UMAEA"]:
                final_emb, weight_norm = self.model.joint_emb_generat(test=test)
            else:
                final_emb = self.model.joint_emb_generat()
                weight_norm = None
            final_emb = F.normalize(final_emb)

        top_k = [1, 10, 50]
        acc_l2r = np.zeros((len(top_k)), dtype=np.float32)
        acc_r2l = np.zeros((len(top_k)), dtype=np.float32)
        test_total, test_loss, mean_l2r, mean_r2l, mrr_l2r, mrr_r2l = 0, 0., 0., 0., 0., 0.
        if self.args.distance == 2:
            distance = pairwise_distances(final_emb[test_left], final_emb[test_right])
        elif self.args.distance == 1:
            distance = torch.FloatTensor(scipy.spatial.distance.cdist(
                final_emb[test_left].cpu().data.numpy(),
                final_emb[test_right].cpu().data.numpy(), metric="cityblock"))

        if self.args.csls is True:
            distance = 1 - csls_sim(1 - distance, self.args.csls_k)

        if last_epoch:
            to_write = []
            test_left_np = test_left.cpu().numpy()
            test_right_np = test_right.cpu().numpy()
            to_write.append(["idx", "rank", "query_id", "gt_id", "ret1", "ret2", "ret3", "v1", "v2", "v3"])
        for idx in range(test_left.shape[0]):
            values, indices = torch.sort(distance[idx, :], descending=False)
            rank = (indices == idx).nonzero(as_tuple=False).squeeze().item()
            mean_l2r += (rank + 1)
            mrr_l2r += 1.0 / (rank + 1)
            for i in range(len(top_k)):
                if rank < top_k[i]:
                    acc_l2r[i] += 1
            if last_epoch:
                indices = indices.cpu().numpy()
                to_write.append([idx, rank, test_left_np[idx], test_right_np[idx], test_right_np[indices[0]], test_right_np[indices[1]],
                                 test_right_np[indices[2]], round(values[0].item(), 4), round(values[1].item(), 4), round(values[2].item(), 4)])
        if last_epoch:
            import csv
            if save_name == "":
                save_name = self.args.model_name
            save_pred_path = osp.join(self.args.data_path, self.args.model_name, f"{save_name}_pred")
            os.makedirs(save_pred_path, exist_ok=True)
            with open(osp.join(save_pred_path, f"{self.args.model_name}_{self.args.data_choice}_{self.args.data_split}_{self.args.data_rate}_ep{self.args.il_start}_pred.txt"), "w") as f:
                wr = csv.writer(f, dialect='excel')
                wr.writerows(to_write)
            if w_normalized is not None:
                with open(osp.join(save_pred_path, f"{self.args.model_name}_{self.args.data_choice}_{self.args.data_split}_{self.args.data_rate}_ep{self.args.il_start}_wight.json"), "w") as fp:
                    json.dump(w_normalized.cpu().tolist(), fp)
            if weight_norm is not None:
                wight_dic = {"all": weight_norm.cpu(), "left": weight_norm[test_left].cpu(), "right": weight_norm[test_right].cpu()}
                with open(osp.join(save_pred_path, f"{self.args.model_name}_{self.args.data_choice}_{self.args.data_split}_{self.args.data_rate}_ep{self.args.il_start}_wight_dic.pkl"), "wb") as fp:
                    pickle.dump(wight_dic, fp)

        for idx in range(test_right.shape[0]):
            _, indices = torch.sort(distance[:, idx], descending=False)
            rank = (indices == idx).nonzero(as_tuple=False).squeeze().item()
            mean_r2l += (rank + 1)
            mrr_r2l += 1.0 / (rank + 1)
            for i in range(len(top_k)):
                if rank < top_k[i]:
                    acc_r2l[i] += 1
        mean_l2r /= test_left.size(0)
        mean_r2l /= test_right.size(0)
        mrr_l2r /= test_left.size(0)
        mrr_r2l /= test_right.size(0)
        for i in range(len(top_k)):
            acc_l2r[i] = round(acc_l2r[i] / test_left.size(0), 4)
            acc_r2l[i] = round(acc_r2l[i] / test_right.size(0), 4)
        gc.collect()
        if not self.args.only_test:
            Loss_out = f", Loss = {self.loss_item:.4f}"
        else:
            Loss_out = ""
            self.epoch = "Test"
            self.early_stop_count = 1

        self.logger.info(f"Ep {self.epoch} | l2r: acc of top {top_k} = {acc_l2r}, mr = {mean_l2r:.3f}, mrr = {mrr_l2r:.3f}{Loss_out}")
        self.logger.info(f"Ep {self.epoch} | r2l: acc of top {top_k} = {acc_r2l}, mr = {mean_r2l:.3f}, mrr = {mrr_r2l:.3f}{Loss_out}")
        self.early_stop_count -= 1
        if not self.args.only_test and mrr_l2r > max(self.loss_log.acc) and not last_epoch:
            self.logger.info(f"Best model update in Ep {self.epoch}: MRR from [{max(self.loss_log.acc)}] --> [{mrr_l2r}] ... ")
            self.loss_log.update_acc(mrr_l2r)
            self.early_stop_count = self.early_stop_init

            self.best_model_wts = copy.deepcopy(self.model.state_dict())

    def _load_model(self, model, model_name=None, first_time_load=False):
        # TODO: path
        if model_name is None:
            model_name = self.args.model_name_save
        save_path = osp.join(self.args.data_path, self.args.model_name, 'save')
        save_path = osp.join(save_path, f'{model_name}.pkl')
        path_00 = save_path.replace('il400_b400', 'il250_b250')
        path_012 = save_path.replace('il250_b250', 'il400_b400')

        if len(model_name) == 0 or not (os.path.exists(path_00) or os.path.exists(path_012)):
            if len(model_name) > 0:
                self.logger.info(f"{model_name}.pkl not exist!!")
            # else:
            self.logger.info("Random init...")
            model.cuda()
            return model

        if os.path.exists(save_path):
            model.load_state_dict(torch.load(save_path, map_location=self.args.device))
        elif os.path.exists(path_012):
            model.load_state_dict(torch.load(path_012, map_location=self.args.device))
            model_name = model_name.replace('il250_b250', 'il400_b400')
        else:
            model.load_state_dict(torch.load(path_00, map_location=self.args.device))
            model_name = model_name.replace('il400_b400', 'il250_b250')

        model.cuda()

        if self.args.stage == 0 and first_time_load:
            self.logger.info(f"loading model [{model_name}.pkl] done! Skip the 0 stage ...")
            self.args.stage = 1
        else:
            self.logger.info(f"loading model [{model_name}.pkl] done!")
        return model

    def _save_model(self, model, input_name=""):
        model_name = self.args.model_name
        save_path = osp.join(self.args.data_path, model_name, 'save')
        os.makedirs(save_path, exist_ok=True)
        if input_name == "":
            input_name = self._save_name_define()
        save_path = osp.join(save_path, f'{input_name}.pkl')
        if model is None:
            return
        torch.save(model.state_dict(), save_path)
        self.logger.info(f"saving [{save_path}] done!")
        return save_path


if __name__ == '__main__':
    cfg = cfg()
    cfg.get_args()
    cfgs = cfg.update_train_configs()
    set_seed(cfgs.random_seed)
    # -----  Init ----------
    torch.multiprocessing.set_sharing_strategy('file_system')
    writer, logger = None, None
    logger = initialize_exp(cfgs)
    logger_path = get_dump_path(cfgs)
    cfgs.time_stamp = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
    comment = f'bath_size={cfgs.batch_size} exp_id={cfgs.exp_id}'
    if not cfgs.no_tensorboard and not cfgs.only_test:
        writer = SummaryWriter(log_dir=os.path.join(logger_path, 'tensorboard', cfgs.time_stamp), comment=comment)

    cfgs.device = torch.device(cfgs.gpu)
    # -----  Begin ----------
    torch.cuda.set_device(cfgs.gpu)
    runner = Runner(cfgs, writer, logger)
    if cfgs.only_test:
        runner.test(last_epoch=False)
    else:
        runner.run()

    # -----  End ----------
    if not cfgs.no_tensorboard and not cfgs.only_test:
        writer.close()
        logger.info("done!")
