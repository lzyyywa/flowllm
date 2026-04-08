from torch.nn.modules.loss import CrossEntropyLoss
import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np

def loss_calu(predict, target, config):
    loss_fn = CrossEntropyLoss()
    # 解包 target
    batch_img, batch_attr, batch_obj, batch_target = target
    batch_attr = batch_attr.cuda()
    batch_obj = batch_obj.cuda()
    batch_target = batch_target.cuda()

    # 🔥 如果 predict 是字典 (走的是最新的 C2C+Flow 轨迹对齐架构)
    if isinstance(predict, dict):
        logits_v = predict.get("logits_v")
        logits_o = predict.get("logits_o")
        logits_c = predict.get("logits_c")
        
        # 1. 基础分类 Loss (注意：必须乘上 cosine_scale)
        loss_v = loss_fn(logits_v * config.cosine_scale, batch_attr)
        loss_o = loss_fn(logits_o * config.cosine_scale, batch_obj)
        loss = loss_v + loss_o
        
        # 安全处理 logits_c (我们在时序流里去掉了它的输出，所以它是 None)
        if logits_c is not None:
            loss_c = loss_fn(logits_c * config.cosine_scale, batch_target)
            loss = loss + loss_c
        
        # 2. 🔥 轨迹对齐 Loss (核心创新点: Flow Matching MSE Loss)
        if "pred_v_v_seq" in predict and "true_v_v_seq" in predict:
            pred_v_seq = predict["pred_v_v_seq"] # [B, 8, 512]
            true_v_seq = predict["true_v_v_seq"] # [B, 8, 512]
            pred_o_seq = predict["pred_v_o_seq"] # [B, 8, 512]
            true_o_seq = predict["true_v_o_seq"] # [B, 8, 512]

            # 动词流和物品流的基础对齐
            loss_flow_v = F.mse_loss(pred_v_seq, true_v_seq)
            loss_flow_o = F.mse_loss(pred_o_seq, true_o_seq)
            loss_flow_total = loss_flow_v + loss_flow_o
            
            # 🔥 端到端 Composer 组合轨迹对齐 (至关重要！替代岭回归)
            if "pred_v_c_seq" in predict and "true_v_c_seq" in predict:
                pred_c_seq = predict["pred_v_c_seq"]
                true_c_seq = predict["true_v_c_seq"]
                loss_flow_c = F.mse_loss(pred_c_seq, true_c_seq)
                loss_flow_total += loss_flow_c
            
            # 使用统一的 flow_loss_weight (与 train_models.py 保持同名和同量级)
            lambda_flow = getattr(config, 'flow_loss_weight', 10.0)
            
            loss = loss + lambda_flow * loss_flow_total
            
        return loss

    # 兼容原版的元组输出逻辑 (以防跑 Vanilla 参照实验)
    else:
        logits, logits_att, logits_obj, logits_soft_prompt = predict
        loss_logit_df = loss_fn(logits, batch_target)
        loss_logit_sp = loss_fn(logits_soft_prompt, batch_target)
        loss_att = loss_fn(logits_att, batch_attr)
        loss_obj = loss_fn(logits_obj, batch_obj)
        loss = loss_logit_df + getattr(config, 'att_obj_w', 1.0) * (loss_att + loss_obj) + getattr(config, 'sp_w', 1.0) * loss_logit_sp
        return loss

class KLLoss(nn.Module):
    def __init__(self, error_metric=nn.KLDivLoss(reduction='mean')):
        super().__init__()
        print('=========using KL Loss=and has temperature and * bz==========')
        self.error_metric = error_metric

    def forward(self, prediction, label, mul=False):
        batch_size = prediction.shape[0]
        probs1 = F.log_softmax(prediction, 1)
        probs2 = F.softmax(label, 1)
        loss = self.error_metric(probs1, probs2)
        if mul:
            return loss * batch_size
        else:
            return loss

def hsic_loss(input1, input2, unbiased=False):
    def _kernel(X, sigma):
        X = X.view(len(X), -1)
        XX = X @ X.t()
        X_sqnorms = torch.diag(XX)
        X_L2 = -2 * XX + X_sqnorms.unsqueeze(1) + X_sqnorms.unsqueeze(0)
        gamma = 1 / (2 * sigma ** 2)
        kernel_XX = torch.exp(-gamma * X_L2)
        return kernel_XX

    N = len(input1)
    if N < 4:
        return torch.tensor(0.0).to(input1.device)
    sigma_x = np.sqrt(input1.size()[1])
    sigma_y = np.sqrt(input2.size()[1])
    kernel_XX = _kernel(input1, sigma_x)
    kernel_YY = _kernel(input2, sigma_y)

    if unbiased:
        tK = kernel_XX - torch.diag(torch.diag(kernel_XX))
        tL = kernel_YY - torch.diag(torch.diag(kernel_YY))
        hsic = (
                torch.trace(tK @ tL)
                + (torch.sum(tK) * torch.sum(tL) / (N - 1) / (N - 2))
                - (2 * torch.sum(tK, 0).dot(torch.sum(tL, 0)) / (N - 2))
        )
        loss = hsic / (N * (N - 3))
    else:
        KH = kernel_XX - kernel_XX.mean(0, keepdim=True)
        LH = kernel_YY - kernel_YY.mean(0, keepdim=True)
        loss = torch.trace(KH @ LH / (N - 1) ** 2)
    return loss

class Gml_loss(nn.Module):
    def __init__(self, error_metric=nn.KLDivLoss(reduction='mean')):
        super().__init__()

    def forward(self, p_o_on_v, v_label, n_c, t=100.0):
        n_c = n_c[:, 0]
        b = p_o_on_v.shape[0]
        n_o = p_o_on_v.shape[-1]
        p_o = p_o_on_v[range(b), v_label, :]  # b,n_o
        num_c = n_c.sum().view(1, -1)  # 1,n_o
        p_o_exp = torch.exp(p_o * t)
        p_o_exp_wed = num_c * p_o_exp  # b,n_o
        p_phi = p_o_exp_wed / (torch.sum(p_o_exp_wed, dim=0, keepdim=True) + 1.0e-6)  # b,n_o
        p_ba = torch.sum(p_phi * n_c, dim=0, keepdim=True) / (num_c + 1.0e-6)  # 1,n_o
        p_ba[torch.where(p_ba < 1.0e-8)] = 1.0
        p_ba_log = torch.log(p_ba)
        loss = (-1.0 / n_o) * p_ba_log.sum()
        return loss