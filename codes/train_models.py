import os
import random
from torch.utils.data.dataloader import DataLoader
import tqdm
import test as test
from loss import *
import torch.multiprocessing
import numpy as np
import json
import torch.nn.functional as F

def cal_conditional(attr2idx, obj2idx, set_name, daset):
    def load_split(path):
        with open(path, 'r') as f:
            loaded_data = json.load(f)
        return loaded_data

    train_data = daset.train_data
    val_data = daset.val_data
    test_data = daset.test_data
    all_data = train_data + val_data + test_data
    if set_name == 'test': used_data = test_data
    elif set_name == 'all': used_data = all_data
    elif set_name == 'train': used_data = train_data

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
    all_logits, all_attr_gt, all_obj_gt, all_pair_gt, loss_avg = test.predict_logits(model, dataset, config)
    test_stats = test.test(dataset, evaluator, all_logits, all_attr_gt, all_obj_gt, all_pair_gt, config)
    result = ""
    key_set = ["attr_acc", "obj_acc", "ub_seen", "ub_unseen", "ub_all", "best_seen", "best_unseen", "best_hm", "AUC"]
    for key in key_set: result = result + key + "  " + str(round(test_stats[key], 4)) + "| "
    print(result)
    model.train()
    return loss_avg, test_stats

def save_checkpoint(state, save_path, epoch, best=False):
    filename = os.path.join(save_path, f"epoch_resume.pt")
    torch.save(state, filename)

def c2c_vanilla(model, optimizer, lr_scheduler, config, train_dataset, val_dataset, test_dataset, scaler):
    train_dataloader = DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=True)
    model.train()
    best_loss = 1e5
    best_metric = 0
    Loss_fn = CrossEntropyLoss()
    log_training = open(os.path.join(config.save_path, 'log.txt'), 'w')

    attr2idx = train_dataset.attr2idx
    obj2idx = train_dataset.obj2idx
    train_pairs = torch.tensor([(attr2idx[attr], obj2idx[obj]) for attr, obj in train_dataset.train_pairs]).cuda()

    for i in range(config.epoch_start, config.epochs):
        progress_bar = tqdm.tqdm(total=len(train_dataloader), desc="epoch % 3d" % (i + 1))
        
        epoch_train_losses = []
        epoch_com_losses = []
        epoch_mse_losses = []
        epoch_comp_losses = []
        epoch_flow_ce_losses = [] 

        use_flow = getattr(config, 'use_flow', False)
        temp_lr = optimizer.param_groups[-1]['lr']
        print(f'Current_lr:{temp_lr}')

        for bid, batch in enumerate(train_dataloader):
            batch_verb = batch[1].cuda()
            batch_obj = batch[2].cuda()
            batch_target = batch[3].cuda()
            batch_img = batch[0].cuda()

            with torch.cuda.amp.autocast(enabled=True):
                if use_flow:
                    outputs = model(batch_img, pairs=train_pairs, verb_labels=batch_verb, obj_labels=batch_obj)

                    loss_verb = Loss_fn(outputs['logits_v'] * config.cosine_scale, batch_verb)
                    loss_obj = Loss_fn(outputs['logits_o'] * config.cosine_scale, batch_obj)
                    loss_com = Loss_fn(outputs['logits_c'] * config.cosine_scale, batch_target)

                    # 基础交叉熵 Loss
                    loss_v_flow = Loss_fn(outputs['logits_v_flow'] * config.cosine_scale, batch_verb)
                    loss_o_flow = Loss_fn(outputs['logits_o_flow'] * config.cosine_scale, batch_obj)
                    loss_c_flow = Loss_fn(outputs['logits_c_flow'] * config.cosine_scale, batch_target)

                    total_flow_ce = loss_v_flow + loss_o_flow + loss_c_flow

                    # 【核心修复 1】：打破扁平化陷阱！使用 sum 恢复真实向量距离
                    loss_mse_total = torch.mean(torch.sum((outputs["pred_v_v"] - outputs["true_v_v"])**2, dim=-1)) + \
                                     torch.mean(torch.sum((outputs["pred_v_o"] - outputs["true_v_o"])**2, dim=-1))

                    with torch.no_grad():
                        delta_v_t = F.normalize(outputs["raw_v_v_0"], dim=-1)
                        delta_o_t = F.normalize(outputs["raw_v_o_0"], dim=-1)

                        A = torch.stack([delta_v_t, delta_o_t], dim=-1).float()
                        B_target = outputs["true_v_c"].unsqueeze(-1).float()

                        A_t = A.transpose(-2, -1)
                        ATA = torch.bmm(A_t, A).float()
                        ATB = torch.bmm(A_t, B_target).float()

                        lambda_ridge = 0.1
                        I = torch.eye(2, device=A.device, dtype=torch.float32).unsqueeze(0)
                        ATA_ridge = ATA + lambda_ridge * I

                        coeffs_star = torch.linalg.solve(ATA_ridge, ATB).squeeze(-1)
                        a_star = coeffs_star[:, 0:1].to(outputs["pred_a"].dtype)
                        b_star = coeffs_star[:, 1:2].to(outputs["pred_b"].dtype)

                    loss_comp = F.mse_loss(outputs["pred_a"], a_star) + F.mse_loss(outputs["pred_b"], b_star)

                    # 【核心修复 2】：终点 MSE 锁死，同样需要恢复量级！
                    pred_x1_norm = F.normalize(outputs["pred_x1_c_0"], dim=-1)
                    target_x1_norm = F.normalize(outputs["target_x1_c"], dim=-1)
                    loss_endpoint_mse = torch.mean(torch.sum((pred_x1_norm - target_x1_norm)**2, dim=-1))

                    flow_weight = getattr(config, 'flow_loss_weight', 1.0)
                    comp_weight = getattr(config, 'composer_weight', 1.0)
                    flow_ce_weight = getattr(config, 'flow_ce_weight', 0.5)

                    loss = loss_com + 0.2 * (loss_verb + loss_obj) + \
                           flow_weight * loss_mse_total + comp_weight * loss_comp + \
                           flow_ce_weight * total_flow_ce + 1.0 * loss_endpoint_mse

                    mse_loss_val = loss_mse_total.item()
                    comp_loss_val = loss_comp.item()
                    flow_ce_val = total_flow_ce.item()

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
                    flow_ce_val = 0.0

                loss = loss / config.gradient_accumulation_steps

            scaler.scale(loss).backward()
            if ((bid + 1) % config.gradient_accumulation_steps == 0) or (bid + 1 == len(train_dataloader)):
                scaler.unscale_(optimizer)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            epoch_train_losses.append(loss.item())
            epoch_com_losses.append(loss_com.item())

            if use_flow:
                epoch_mse_losses.append(mse_loss_val)
                epoch_comp_losses.append(comp_loss_val)
                epoch_flow_ce_losses.append(flow_ce_val)

            postfix_dict = { "loss": np.mean(epoch_train_losses[-50:]), "l_com": np.mean(epoch_com_losses[-50:]) }
            if use_flow:
                postfix_dict["l_mse"] = np.mean(epoch_mse_losses[-50:])
                postfix_dict["l_flow_ce"] = np.mean(epoch_flow_ce_losses[-50:])
            progress_bar.set_postfix(postfix_dict)
            progress_bar.update()

        lr_scheduler.step()
        progress_bar.close()

        epoch_str = f"epoch {i + 1} loss: {np.mean(epoch_train_losses):.4f}, loss_com: {np.mean(epoch_com_losses):.4f}"
        if use_flow: epoch_str += f", mse: {np.mean(epoch_mse_losses):.4f}, flow_ce: {np.mean(epoch_flow_ce_losses):.4f}"

        progress_bar.write(epoch_str)
        log_training.write(f"{epoch_str}\n")

        if (i + 1) % config.save_every_n == 0:
            save_checkpoint({'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': lr_scheduler.state_dict(), 'scaler': scaler.state_dict()}, config.save_path, i)

        key_set = ["attr_acc", "obj_acc", "ub_seen", "ub_unseen", "ub_all", "best_seen", "best_unseen", "best_hm", "AUC"]
        if i % config.eval_every_n == 0 or i + 1 == config.epochs or i >= config.val_epochs_ts:
            print("Evaluating val dataset:")
            loss_avg, val_result = evaluate(model, val_dataset, config)
            result = ""
            for key in val_result:
                if key in key_set: result = result + key + "  " + str(round(val_result[key], 4)) + "| "
            log_training.write(result + '\n')

            if config.best_model_metric == "best_loss":
                if loss_avg.cpu().float() < best_loss:
                    best_loss = loss_avg.cpu().float()
                    loss_avg, val_result = evaluate(model, test_dataset, config)
                    torch.save(model.state_dict(), os.path.join(config.save_path, "best.pt"))
            else:
                if val_result[config.best_model_metric] > best_metric:
                    best_metric = val_result[config.best_model_metric]
                    loss_avg, val_result = evaluate(model, test_dataset, config)
                    torch.save(model.state_dict(), os.path.join(config.save_path, "best.pt"))
        log_training.flush()
        if i + 1 == config.epochs:
            model.load_state_dict(torch.load(os.path.join(config.save_path, "best.pt")))
            loss_avg, val_result = evaluate(model, test_dataset, config)

def c2c_enhance(model, optimizer, lr_scheduler, config, train_dataset, val_dataset, test_dataset, scaler):
    pass