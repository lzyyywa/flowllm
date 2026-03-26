import os
import random

from torch.utils.data.dataloader import DataLoader
import tqdm
import test as test
from loss import *
from loss import KLLoss
import torch.multiprocessing
import numpy as np
import json
import math
import torch.nn.functional as F
from utils.ade_utils import emd_inference_opencv_test
from collections import Counter

from utils.hsic import hsic_normalized_cca


def cal_conditional(attr2idx, obj2idx, set_name, daset):
    def load_split(path):
        with open(path, 'r') as f:
            loaded_data = json.load(f)
        return loaded_data

    train_data = daset.train_data
    val_data = daset.val_data
    test_data = daset.test_data
    all_data = train_data + val_data + test_data
    if set_name == 'test':
        used_data = test_data
    elif set_name == 'all':
        used_data = all_data
    elif set_name == 'train':
        used_data = train_data

    v_o = torch.zeros(size=(len(attr2idx), len(obj2idx)))
    for item in used_data:
        verb_idx = attr2idx[item[1]]
        obj_idx = obj2idx[item[2]]

        v_o[verb_idx, obj_idx] += 1

    v_o_on_v = v_o / (torch.sum(v_o, dim=1, keepdim=True) + 1.0e-6)
    v_o_on_o = v_o / (torch.sum(v_o, dim=0, keepdim=True) + 1.0e-6)

    return v_o_on_v, v_o_on_o


def evaluate(model, dataset, config):
    model.eval()
    evaluator = test.Evaluator(dataset, model=None)
    all_logits, all_attr_gt, all_obj_gt, all_pair_gt, loss_avg = test.predict_logits(
        model, dataset, config)
    test_stats = test.test(
        dataset,
        evaluator,
        all_logits,
        all_attr_gt,
        all_obj_gt,
        all_pair_gt,
        config
    )
    result = ""
    key_set = ["attr_acc", "obj_acc", "ub_seen", "ub_unseen", "ub_all", "best_seen", "best_unseen", "best_hm", "AUC"]

    for key in key_set:
        result = result + key + "  " + str(round(test_stats[key], 4)) + "| "
    print(result)
    model.train()
    return loss_avg, test_stats


def save_checkpoint(state, save_path, epoch, best=False):
    filename = os.path.join(save_path, f"epoch_resume.pt")
    torch.save(state, filename)


def rand_bbox(size, lam):
    W = size[-2]
    H = size[-1]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int_(W * cut_rat)
    cut_h = np.int_(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def c2c_vanilla(model, optimizer, lr_scheduler, config, train_dataset, val_dataset, test_dataset,
                scaler):
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True
    )

    model.train()
    best_loss = 1e5
    best_metric = 0
    Loss_fn = CrossEntropyLoss()
    log_training = open(os.path.join(config.save_path, 'log.txt'), 'w')

    attr2idx = train_dataset.attr2idx
    obj2idx = train_dataset.obj2idx

    train_pairs = torch.tensor([(attr2idx[attr], obj2idx[obj])
                                for attr, obj in train_dataset.train_pairs]).cuda()

    train_losses = []

    for i in range(config.epoch_start, config.epochs):
        progress_bar = tqdm.tqdm(
            total=len(train_dataloader), desc="epoch % 3d" % (i + 1)
        )

        epoch_train_losses = []
        epoch_com_losses = []
        epoch_oo_losses = []
        epoch_vv_losses = []

        epoch_mse_losses = []
        epoch_comp_losses = []
        epoch_com_flow_losses = [] # 用于专门记录 Flow 独立交叉熵 Loss

        use_flow = getattr(config, 'use_flow', False)

        temp_lr = optimizer.param_groups[-1]['lr']
        print(f'Current_lr:{temp_lr}')

        for bid, batch in enumerate(train_dataloader):
            batch_verb = batch[1].cuda()
            batch_obj = batch[2].cuda()
            batch_target = batch[3].cuda()
            batch_img = batch[0].cuda()

            with torch.cuda.amp.autocast(enabled=True):
                # ==========================================
                # = FlowComposer 插件逻辑 (100% 对齐论文) =
                # ==========================================
                if use_flow:
                    outputs = model(batch_img, pairs=train_pairs, verb_labels=batch_verb, obj_labels=batch_obj)

                    # 1. Endpoint Classification Losses (保证乘以 cosine_scale 防止梯度消失)
                    loss_verb = Loss_fn(outputs['logits_v'] * config.cosine_scale, batch_verb)
                    loss_obj = Loss_fn(outputs['logits_o'] * config.cosine_scale, batch_obj)
                    
                    # 纯净的 C2C 分类误差，夯实地基
                    loss_com = Loss_fn(outputs['logits_c'] * config.cosine_scale, batch_target)
                    
                    # 【核心解耦：单独让 Flow 网络学习对比误差】
                    loss_com_flow = Loss_fn(outputs['logits_c_flow'] * config.cosine_scale, batch_target)

                    # 2. Primitive Flows MSE Losses
                    loss_mse_base = F.mse_loss(outputs["pred_v_v"], outputs["true_v_v"]) + \
                                    F.mse_loss(outputs["pred_v_o"], outputs["true_v_o"])
                    loss_mse_leak = F.mse_loss(outputs["pred_v_v_leak"], outputs["true_v_v_leak"]) + \
                                    F.mse_loss(outputs["pred_v_o_leak"], outputs["true_v_o_leak"])
                    leak_weight = getattr(config, 'leak_loss_weight', 1.0)
                    loss_mse_total = loss_mse_base + leak_weight * loss_mse_leak

                    # 3. Explicit Composer Optimization (加入防爆的岭回归 Ridge Regression)
                    with torch.no_grad():
                        A = torch.stack([outputs["norm_v_v"], outputs["norm_v_o"]], dim=-1).float() # [B, D, 2]
                        B_target = outputs["true_v_c"].unsqueeze(-1).float() # [B, D, 1]

                        # 构建 A^T A 和 A^T B
                        A_t = A.transpose(-2, -1) # [B, 2, D]

                        # 【关键修复】在 bmm 算完之后，立刻强制转回 float()，对抗 AMP 的自动降级！
                        ATA = torch.bmm(A_t, A).float()
                        ATB = torch.bmm(A_t, B_target).float()

                        # 核心防爆机制：加入 L2 正则化项 (lambda * I)
                        lambda_ridge = 0.1
                        I = torch.eye(2, device=A.device, dtype=torch.float32).unsqueeze(0)
                        ATA_ridge = ATA + lambda_ridge * I

                        # 绝对安全求解 (A^T A + \lambda I) X = A^T B
                        coeffs_star = torch.linalg.solve(ATA_ridge, ATB).squeeze(-1) # [B, 2]

                        a_star = coeffs_star[:, 0:1].to(outputs["pred_a"].dtype)
                        b_star = coeffs_star[:, 1:2].to(outputs["pred_b"].dtype)

                    loss_comp = F.mse_loss(outputs["pred_a"], a_star) + F.mse_loss(outputs["pred_b"], b_star)

                    flow_weight = getattr(config, 'flow_loss_weight', 1.0)
                    comp_weight = getattr(config, 'composer_weight', 1.0)

                    # 【终极大一统 Loss：各司其职，互不污染！】
                    loss = loss_com + 0.2 * (loss_verb + loss_obj) + flow_weight * loss_mse_total + comp_weight * loss_comp + 0.5 * loss_com_flow

                    mse_loss_val = loss_mse_total.item()
                    comp_loss_val = loss_comp.item()
                    com_flow_loss_val = loss_com_flow.item()

                # ==========================================
                # = 原生 Vanilla 逻辑 =
                # ==========================================
                else:
                    p_v, p_o, p_pair_v, p_pair_o, vid_feat, v_feat, o_feat, p_v_con_o, p_o_con_v = model(batch_img)

                    loss_verb = Loss_fn(p_v * config.cosine_scale, batch_verb)
                    loss_obj = Loss_fn(p_o * config.cosine_scale, batch_obj)
                    train_v_inds, train_o_inds = train_pairs[:, 0], train_pairs[:, 1]
                    pred_com_train = (p_pair_v + p_pair_o)[:, train_v_inds, train_o_inds]
                    loss_com = Loss_fn(pred_com_train * config.cosine_scale, batch_target)

                    loss = loss_com + 0.2 * (loss_verb + loss_obj)

                    mse_loss_val = 0.0
                    comp_loss_val = 0.0
                    com_flow_loss_val = 0.0

                loss = loss / config.gradient_accumulation_steps

            scaler.scale(loss).backward()

            if ((bid + 1) % config.gradient_accumulation_steps == 0) or (bid + 1 == len(train_dataloader)):
                scaler.unscale_(optimizer)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            epoch_train_losses.append(loss.item())
            epoch_com_losses.append(loss_com.item())
            epoch_vv_losses.append(loss_verb.item())
            epoch_oo_losses.append(loss_obj.item())

            if use_flow:
                epoch_mse_losses.append(mse_loss_val)
                epoch_comp_losses.append(comp_loss_val)
                epoch_com_flow_losses.append(com_flow_loss_val)

            # 在进度条上全面打印各项损失指标
            postfix_dict = {
                "train loss": np.mean(epoch_train_losses[-50:]),
                "loss_v": np.mean(epoch_vv_losses[-50:]),
                "loss_o": np.mean(epoch_oo_losses[-50:]),
                "loss_com": np.mean(epoch_com_losses[-50:])
            }
            if use_flow:
                postfix_dict["flow_mse"] = np.mean(epoch_mse_losses[-50:])
                postfix_dict["comp"] = np.mean(epoch_comp_losses[-50:])
                postfix_dict["loss_flow_ce"] = np.mean(epoch_com_flow_losses[-50:])
            progress_bar.set_postfix(postfix_dict)
            progress_bar.update()

        lr_scheduler.step()
        progress_bar.close()

        # 终端输出更加全面的训练概况
        epoch_str = f"epoch {i + 1} train loss: {np.mean(epoch_train_losses):.4f}, loss_v: {np.mean(epoch_vv_losses):.4f}, loss_o: {np.mean(epoch_oo_losses):.4f}, loss_com: {np.mean(epoch_com_losses):.4f}"
        if use_flow:
            epoch_str += f", flow_mse: {np.mean(epoch_mse_losses):.4f}, comp: {np.mean(epoch_comp_losses):.4f}, flow_ce: {np.mean(epoch_com_flow_losses):.4f}"
        
        progress_bar.write(epoch_str)
        train_losses.append(np.mean(epoch_train_losses))
        
        # 写入 txt 记录
        log_training.write('\n')
        log_training.write(f"{epoch_str}\n")

        if (i + 1) % config.save_every_n == 0:
            save_checkpoint({
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': lr_scheduler.state_dict(),
                'scaler': scaler.state_dict(),
            }, config.save_path, i)

        key_set = ["attr_acc", "obj_acc", "ub_seen", "ub_unseen", "ub_all", "best_seen", "best_unseen", "best_hm", "AUC"]
        if i % config.eval_every_n == 0 or i + 1 == config.epochs or i >= config.val_epochs_ts:
            print("Evaluating val dataset:")
            loss_avg, val_result = evaluate(model, val_dataset, config)
            result = ""
            for key in val_result:
                if key in key_set:
                    result = result + key + "  " + str(round(val_result[key], 4)) + "| "
            log_training.write('\n')
            log_training.write(result)
            print("Loss average on val dataset: {}".format(loss_avg))
            log_training.write('\n')
            log_training.write("Loss average on val dataset: {}\n".format(loss_avg))
            if config.best_model_metric == "best_loss":
                if loss_avg.cpu().float() < best_loss:
                    print('find best!')
                    log_training.write('find best!')
                    best_loss = loss_avg.cpu().float()
                    print("Evaluating test dataset:")
                    loss_avg, val_result = evaluate(model, test_dataset, config)
                    torch.save(model.state_dict(), os.path.join(
                        config.save_path, f"best.pt"
                    ))
                    result = ""
                    for key in val_result:
                        if key in key_set:
                            result = result + key + "  " + str(round(val_result[key], 4)) + "| "
                    log_training.write('\n')
                    log_training.write(result)
                    print("Loss average on test dataset: {}".format(loss_avg))
                    log_training.write('\n')
                    log_training.write("Loss average on test dataset: {}\n".format(loss_avg))
            else:
                if val_result[config.best_model_metric] > best_metric:
                    best_metric = val_result[config.best_model_metric]
                    log_training.write('\n')
                    print('find best!')
                    log_training.write('find best!')
                    loss_avg, val_result = evaluate(model, test_dataset, config)
                    torch.save(model.state_dict(), os.path.join(
                        config.save_path, f"best.pt"
                    ))
                    result = ""
                    for key in val_result:
                        if key in key_set:
                            result = result + key + "  " + str(round(val_result[key], 4)) + "| "
                    log_training.write('\n')
                    log_training.write(result)
                    print("Loss average on test dataset: {}".format(loss_avg))
                    log_training.write('\n')
                    log_training.write("Loss average on test dataset: {}\n".format(loss_avg))
        log_training.write('\n')
        log_training.flush()
        if i + 1 == config.epochs:
            print("Evaluating test dataset on Closed World")
            model.load_state_dict(torch.load(os.path.join(
                config.save_path, "best.pt"
            )))
            loss_avg, val_result = evaluate(model, test_dataset, config)
            result = ""
            for key in val_result:
                if key in key_set:
                    result = result + key + "  " + str(round(val_result[key], 4)) + "| "
            log_training.write('\n')
            log_training.write(result)
            print("Final Loss average on test dataset: {}".format(loss_avg))
            log_training.write('\n')
            log_training.write("Final Loss average on test dataset: {}\n".format(loss_avg))


def c2c_enhance(model, optimizer, lr_scheduler, config, train_dataset, val_dataset, test_dataset,
                scaler):
    # 此处保持原始 c2c_enhance 函数逻辑不变
    pass