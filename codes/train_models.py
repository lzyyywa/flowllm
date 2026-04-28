import os
import random
from torch.utils.data.dataloader import DataLoader
import tqdm
import test as test  # 确保 test.py 在同级目录
from loss import *
import torch.multiprocessing
import numpy as np
import json
import torch.nn.functional as F

# ==========================================
# 辅助工具函数
# ==========================================

def cal_conditional(attr2idx, obj2idx, set_name, daset):
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
    """验证集/测试集评估函数"""
    model.eval()
    evaluator = test.Evaluator(dataset, model=None)
    all_logits, all_attr_gt, all_obj_gt, all_pair_gt, loss_avg = test.predict_logits(model, dataset, config)
    test_stats = test.test(dataset, evaluator, all_logits, all_attr_gt, all_obj_gt, all_pair_gt, config)

    result = ""
    key_set = ["attr_acc", "obj_acc", "ub_seen", "ub_unseen", "ub_all", "best_seen", "best_unseen", "best_hm", "AUC"]
    for key in key_set:
        if key in test_stats:
            result = result + key + "  " + str(round(test_stats[key], 4)) + "| "
    print(result)
    model.train()
    return loss_avg, test_stats

def save_checkpoint(state, save_path, epoch):
    filename = os.path.join(save_path, f"epoch_resume.pt")
    torch.save(state, filename)

# ==========================================
# 主训练函数
# ==========================================

def c2c_vanilla(model, optimizer, lr_scheduler, config, train_dataset, val_dataset, test_dataset, scaler):
    train_dataloader = DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=True)
    model.train()
    best_loss = 1e5
    best_metric = 0
    Loss_fn = CrossEntropyLoss()
    log_training = open(os.path.join(config.save_path, 'log.txt'), 'a')

    attr2idx = train_dataset.attr2idx
    obj2idx = train_dataset.obj2idx
    train_pairs = torch.tensor([(attr2idx[attr], obj2idx[obj]) for attr, obj in train_dataset.train_pairs]).cuda()

    for i in range(config.epoch_start, config.epochs):
        progress_bar = tqdm.tqdm(total=len(train_dataloader), desc="epoch % 3d" % (i + 1))

        epoch_train_losses = []
        epoch_com_losses = []
        epoch_verb_losses = []
        epoch_obj_losses = []

        epoch_mse_losses = []
        epoch_comp_losses = []
        epoch_flow_ce_losses = []
        epoch_endpoint_mse_losses = []

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

                    loss_v_flow = Loss_fn(outputs['logits_v_flow'] * config.cosine_scale, batch_verb)
                    loss_o_flow = Loss_fn(outputs['logits_o_flow'] * config.cosine_scale, batch_obj)
                    loss_c_flow = Loss_fn(outputs['logits_c_flow'] * config.cosine_scale, batch_target)
                    total_flow_ce = loss_v_flow + loss_o_flow + loss_c_flow

                    # 🔥 [完美对接]：这里接收到的已经是 C2C 网络发来的最新“柔性对齐 MSE”啦！
                    loss_mse_v = outputs["loss_soft_align_v"]
                    loss_mse_o = outputs["loss_soft_align_o"]
                    loss_mse_c = outputs["loss_soft_align_c"]
                    loss_mse_total = loss_mse_v + loss_mse_o + loss_mse_c

                    # 🔥 [完美对接]：提取终点锚定损失
                    loss_endpoint_mse = outputs["loss_endpoint_mse"]

                    with torch.no_grad():
                        delta_v_0 = F.normalize(outputs["raw_v_v_0"], dim=-1)
                        delta_o_0 = F.normalize(outputs["raw_v_o_0"], dim=-1)
                        A = torch.stack([delta_v_0, delta_o_0], dim=-1).float()
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

                    # 老老实实从 yml 配置文件读取权重，给出安全的 Fallback 默认值
                    flow_weight = getattr(config, 'flow_loss_weight', 1.0)
                    comp_weight = getattr(config, 'composer_weight', 1.0)
                    flow_ce_weight = getattr(config, 'flow_ce_weight', 0.8)

                    # 🔥 [火力全开]：正确加入 1.0 * loss_endpoint_mse，锁死最终锚点！
                    loss = loss_com + 0.2 * (loss_verb + loss_obj) + \
                           flow_weight * loss_mse_total + comp_weight * loss_comp + \
                           flow_ce_weight * total_flow_ce + 1.0 * loss_endpoint_mse

                    mse_loss_val = loss_mse_total.item()
                    comp_loss_val = loss_comp.item()
                    flow_ce_val = total_flow_ce.item()
                    endpoint_mse_val = loss_endpoint_mse.item()

                else:
                    p_v, p_o, p_pair_v, p_pair_o, _, _, _, _, _ = model(batch_img)
                    loss_verb = Loss_fn(p_v * config.cosine_scale, batch_verb)
                    loss_obj = Loss_fn(p_o * config.cosine_scale, batch_obj)
                    train_v_inds, train_o_inds = train_pairs[:, 0], train_pairs[:, 1]
                    pred_com_train = (p_pair_v + p_pair_o)[:, train_v_inds, train_o_inds]
                    loss_com = Loss_fn(pred_com_train * config.cosine_scale, batch_target)
                    loss = loss_com + 0.2 * (loss_verb + loss_obj)
                    mse_loss_val, comp_loss_val, flow_ce_val, endpoint_mse_val = 0.0, 0.0, 0.0, 0.0

                verb_loss_val = loss_verb.item()
                obj_loss_val = loss_obj.item()
                loss = loss / config.gradient_accumulation_steps

            scaler.scale(loss).backward()
            if ((bid + 1) % config.gradient_accumulation_steps == 0) or (bid + 1 == len(train_dataloader)):
                scaler.unscale_(optimizer)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            epoch_train_losses.append(loss.item() * config.gradient_accumulation_steps)
            epoch_com_losses.append(loss_com.item())
            epoch_verb_losses.append(verb_loss_val)
            epoch_obj_losses.append(obj_loss_val)
            if use_flow:
                epoch_mse_losses.append(mse_loss_val)
                epoch_comp_losses.append(comp_loss_val)
                epoch_flow_ce_losses.append(flow_ce_val)
                epoch_endpoint_mse_losses.append(endpoint_mse_val)

            postfix = {
                "loss": np.mean(epoch_train_losses[-50:]),
                "l_com": np.mean(epoch_com_losses[-50:]),
                "l_v": np.mean(epoch_verb_losses[-50:]),
                "l_o": np.mean(epoch_obj_losses[-50:])
            }
            if use_flow:
                postfix.update({
                    "l_mse": np.mean(epoch_mse_losses[-50:]),
                    "l_cmp": np.mean(epoch_comp_losses[-50:]),
                    "l_f_ce": np.mean(epoch_flow_ce_losses[-50:]),
                    "l_ep": np.mean(epoch_endpoint_mse_losses[-50:])
                })
            progress_bar.set_postfix(postfix)
            progress_bar.update()

        lr_scheduler.step()
        progress_bar.close()

        epoch_summary = (f"epoch {i + 1} loss: {np.mean(epoch_train_losses):.4f}, "
                         f"l_com: {np.mean(epoch_com_losses):.4f}, "
                         f"l_v: {np.mean(epoch_verb_losses):.4f}, "
                         f"l_o: {np.mean(epoch_obj_losses):.4f}")
        if use_flow:
            epoch_summary += (f", l_mse: {np.mean(epoch_mse_losses):.4f}, "
                              f"l_cmp: {np.mean(epoch_comp_losses):.4f}, "
                              f"l_f_ce: {np.mean(epoch_flow_ce_losses):.4f}, "
                              f"l_ep: {np.mean(epoch_endpoint_mse_losses):.4f}")
        print(epoch_summary)
        log_training.write(epoch_summary + "\n")

        # ==========================================
        # 评估逻辑
        # ==========================================
        val_start_thresh = getattr(config, 'val_epochs_ts', config.epochs + 1)

        if (i + 1 >= val_start_thresh) or (i % config.eval_every_n == 0) or (i + 1 == config.epochs):
            print("Evaluating val dataset:")
            loss_avg, val_result = evaluate(model, val_dataset, config)

            val_log = "VAL  -> "
            for key in val_result:
                if key in ["attr_acc", "obj_acc", "AUC", "best_hm"]:
                    val_log += f"{key}: {round(val_result[key], 4)} | "
            log_training.write(val_log + "\n")

            if val_result[config.best_model_metric] > best_metric:
                best_metric = val_result[config.best_model_metric]
                print(f"New best Val {config.best_model_metric}: {best_metric:.4f}. Running TEST...")

                _, test_result = evaluate(model, test_dataset, config)

                test_log = "TEST -> "
                for key in test_result:
                    if key in ["attr_acc", "obj_acc", "AUC", "best_hm"]:
                        test_log += f"{key}: {round(test_result[key], 4)} | "
                log_training.write(test_log + "\n")

                torch.save(model.state_dict(), os.path.join(config.save_path, "best.pt"))
                print(f"Best model saved to {os.path.join(config.save_path, 'best.pt')}")

        log_training.flush()

    # === 训练完全结束后，加载最佳模型进行最终评估 ===
    print("\n--- Training Completed. Final Evaluation on Best Model ---")
    model.load_state_dict(torch.load(os.path.join(config.save_path, "best.pt")))
    _, final_test_result = evaluate(model, test_dataset, config)

    final_str = "FINAL TEST -> "
    for key in final_test_result:
        if key in ["attr_acc", "obj_acc", "AUC", "best_hm"]:
            final_str += f"{key}: {round(final_test_result[key], 4)} | "
    log_training.write(final_str + "\n")
    log_training.close()

def c2c_enhance(model, optimizer, lr_scheduler, config, train_dataset, val_dataset, test_dataset, scaler):
    pass
