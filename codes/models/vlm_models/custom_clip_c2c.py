import torch
import torch.nn as nn

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from models.vlm_models.text_learner import get_text_learner
import torch.nn.functional as F
from einops import rearrange

_tokenizer = _Tokenizer()

class FlowMLP(nn.Module):
    def __init__(self, feature_dim):
        super(FlowMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim + 1, feature_dim * 2),
            nn.LayerNorm(feature_dim * 2),
            nn.GELU(),
            nn.Linear(feature_dim * 2, feature_dim)
        )

    def forward(self, x, t):
        x_t = torch.cat([x, t], dim=-1)
        return self.net(x_t)

class FlowComposer(nn.Module):
    def __init__(self, feature_dim):
        super(FlowComposer, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, 2)
        )

    def forward(self, v_v, v_o):
        x = torch.cat([v_v, v_o], dim=-1)
        coeffs = self.net(x)
        return coeffs[:, 0:1], coeffs[:, 1:2]

class MLP(nn.Module):
    def __init__(self, inp_dim, out_dim, num_layers=1, relu=True, bias=True, dropout=False, norm=False, layers=[]):
        super(MLP, self).__init__()
        mod = []
        incoming = inp_dim
        for layer_ind in range(num_layers - 1):
            if len(layers) == 0:
                outgoing = incoming
            else:
                outgoing = layers[layer_ind]
            mod.append(nn.Linear(incoming, outgoing, bias=bias))

            incoming = outgoing
            if norm:
                mod.append(nn.LayerNorm(outgoing))
            mod.append(nn.ReLU(inplace=True))
            if dropout:
                mod.append(nn.Dropout(p=0.5))

        mod.append(nn.Linear(incoming, out_dim, bias=bias))

        if relu:
            mod.append(nn.ReLU(inplace=True))
        self.mod = nn.Sequential(*mod)

    def forward(self, x):
        return self.mod(x)


class MLP_ST(nn.Module):
    def __init__(self, inp_dim, out_dim, num_layers=1, relu=True, bias=True, dropout=False, norm=False, layers=[]):
        super(MLP_ST, self).__init__()
        mod = []
        incoming = inp_dim
        for layer_ind in range(num_layers - 1):
            if len(layers) == 0:
                outgoing = incoming
            else:
                outgoing = layers[layer_ind]
            mod.append(nn.Conv1d(incoming, outgoing, kernel_size=3, bias=bias, padding=1))

            incoming = outgoing
            if norm:
                mod.append(nn.LayerNorm(outgoing))
            mod.append(nn.ReLU(inplace=True))
            if dropout:
                mod.append(nn.Dropout(p=0.5))

        mod.append(nn.Conv1d(incoming, out_dim, kernel_size=3, bias=bias, padding=1))

        if relu:
            mod.append(nn.ReLU(inplace=True))
        self.mod = nn.Sequential(*mod)

    def forward(self, x):
        for o in self.mod:
            if isinstance(o, nn.LayerNorm):
                x = x.transpose(1, 2)
                x = o(x)
                x = x.transpose(1, 2)
            else:
                x = o(x)
        return x


class TextEncoder(nn.Module):
    def __init__(self, cfg, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        for block in self.transformer.resblocks:
            block.attn_mask = block.attn_mask[:cfg.ctx_length, :cfg.ctx_length]
        self.dtype = clip_model.dtype

    def forward(self, x, tokenized_prompts):
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x)

        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x


class VideoEncoder(nn.Module):
    def __init__(self, cfg, clip_model):
        super().__init__()
        from models.vlm_models.AIM import get_aim
        self.visual = get_aim(cfg)
        self.clip_proj = clip_model.visual.proj
        self.num_frames=cfg.num_frames

    def forward(self, x):
        out = self.visual(x)
        if self.clip_proj is not None:
            out = out @ self.clip_proj
        out = rearrange(out, '(b t) d -> b d t', t=self.num_frames)
        return out


class CustomCLIP(nn.Module):
    def __init__(self, cfg, train_dataset, clip_model):
        super().__init__()
        self.verb_prompt_learner = get_text_learner(cfg, train_dataset, clip_model, 'verb')
        self.verb_tokenized_prompts = self.verb_prompt_learner.token_ids
        self.obj_prompt_learner = get_text_learner(cfg, train_dataset, clip_model, 'object')
        self.obj_tokenized_prompts = self.obj_prompt_learner.token_ids

        self.text_encoder = TextEncoder(cfg, clip_model)
        self.video_encoder = VideoEncoder(cfg, clip_model)
        self.logit_scale = clip_model.logit_scale

        try:
            fc_emb = cfg.fc_emb.split(',')
        except:
            fc_emb = [cfg.fc_emb]
        layers = []
        for a in fc_emb:
            a = int(a)
            layers.append(a)

        self.c2c_OE1 = MLP(cfg.feat_dim, int(cfg.emb_dim), relu=cfg.relu, num_layers=cfg.nlayers,
                           dropout=False, norm=True, layers=layers)
        self.c2c_OE2 = MLP(cfg.feat_dim, int(cfg.emb_dim), relu=cfg.relu, num_layers=cfg.nlayers,
                           dropout=False, norm=True, layers=layers)
        self.c2c_VE1 = MLP_ST(cfg.feat_dim, int(cfg.emb_dim), relu=cfg.relu, num_layers=cfg.nlayers,
                              dropout=False, norm=True, layers=layers)
        self.c2c_VE2 = MLP_ST(cfg.feat_dim, int(cfg.emb_dim), relu=cfg.relu, num_layers=cfg.nlayers,
                              dropout=False, norm=True, layers=layers)

        self.c2c_f_v_e_o_com = nn.Linear(2 * cfg.emb_dim, cfg.emb_dim, bias=True)
        self.c2c_f_o_e_v_com = nn.Linear(2 * cfg.emb_dim, cfg.emb_dim, bias=True)

        self.c2c_text_v = nn.Linear(cfg.feat_dim, cfg.emb_dim, bias=True)
        self.c2c_text_o = nn.Linear(cfg.feat_dim, cfg.emb_dim, bias=True)

        self.use_flow = getattr(cfg, 'use_flow', False)
        if self.use_flow:
            self.v_flow = FlowMLP(cfg.emb_dim)
            self.o_flow = FlowMLP(cfg.emb_dim)
            self.composer = FlowComposer(cfg.emb_dim)


    def forward(self, video, pairs=None, verb_labels=None, obj_labels=None):
        verb_prompts = self.verb_prompt_learner()
        verb_text_features = self.text_encoder(verb_prompts, self.verb_tokenized_prompts)
        verb_text_features = self.c2c_text_v(verb_text_features)

        obj_prompts = self.obj_prompt_learner()
        obj_text_features = self.text_encoder(obj_prompts, self.obj_tokenized_prompts)
        obj_text_features = self.c2c_text_o(obj_text_features)

        video_features = self.video_encoder(video)

        o_feat = self.c2c_OE1(video_features.mean(dim=-1))
        v_feat_t = self.c2c_VE1(video_features)
        v_feat = v_feat_t.mean(dim=-1)

        if not self.use_flow:
            o_feat_normed = F.normalize(o_feat, dim=1)
            v_feat_normed = F.normalize(v_feat, dim=1)
            verb_text_features_norm = F.normalize(verb_text_features, dim=-1)
            obj_text_features_norm = F.normalize(obj_text_features, dim=-1)
            verb_logits = v_feat_normed @ verb_text_features_norm.t()
            obj_logits = o_feat_normed @ obj_text_features_norm.t()
            verb_logits = verb_logits * 0.5 + 0.5
            obj_logits = obj_logits * 0.5 + 0.5
            b = video_features.shape[0]
            c = verb_text_features.shape[-1]
            n_v = verb_logits.shape[-1]
            n_o = obj_logits.shape[-1]
            o_feat_c = self.c2c_OE2(video_features.mean(dim=-1))
            v_feat_c = self.c2c_VE2(video_features)
            v_feat_c = v_feat_c.mean(dim=-1)
            p_v_con_o, p_o_con_v = self.condition_module(v_feat_c, o_feat_c, verb_text_features, obj_text_features, n_o, b, c, n_v)
            p_pair_o = p_v_con_o * obj_logits.unsqueeze(1)
            p_pair_v = p_o_con_v * verb_logits.unsqueeze(-1)

            if self.training:
                return verb_logits, obj_logits, p_pair_v, p_pair_o, video_features, o_feat, v_feat, p_v_con_o, p_o_con_v
            else:
                verb_idx, obj_idx = pairs[:, 0], pairs[:, 1]
                com_logits = p_pair_o[:, verb_idx, obj_idx] + p_pair_v[:, verb_idx, obj_idx]
                return com_logits

        # =========================================================
        # === FlowComposer Plugin Branch (The True Two-Stream!) ===
        # =========================================================
        else:
            B, D = v_feat.shape
            device = video.device

            # 1. 基础特征提取与 L2 归一化
            x0_v = F.normalize(v_feat, dim=-1)
            x0_o = F.normalize(o_feat, dim=-1)

            o_feat_c = self.c2c_OE2(video_features.mean(dim=-1))
            v_feat_c = self.c2c_VE2(video_features).mean(dim=-1)

            x0_c = F.normalize(v_feat_c + o_feat_c, dim=-1)

            verb_text_features_norm = F.normalize(verb_text_features, dim=-1)
            obj_text_features_norm = F.normalize(obj_text_features, dim=-1)

            # -------------------------------------------------------------
            # 🔥 主路：纯净 C2C 保底逻辑
            # -------------------------------------------------------------
            logits_v_base = x0_v @ verb_text_features_norm.t()
            logits_o_base = x0_o @ obj_text_features_norm.t()

            logits_v_base = logits_v_base * 0.5 + 0.5
            logits_o_base = logits_o_base * 0.5 + 0.5

            c = verb_text_features.shape[-1]
            n_v = logits_v_base.shape[-1]
            n_o = logits_o_base.shape[-1]

            p_v_con_o, p_o_con_v = self.condition_module(v_feat_c, o_feat_c, verb_text_features, obj_text_features, n_o, B, c, n_v)
            p_pair_o = p_v_con_o * logits_o_base.unsqueeze(1)
            p_pair_v = p_o_con_v * logits_v_base.unsqueeze(-1)

            if self.training:
                if verb_labels is None or obj_labels is None:
                    raise ValueError("Flow training requires `verb_labels` and `obj_labels`.")

                # 🛡️ 【神级防御：.detach() 单向玻璃墙】
                x0_v_f = x0_v.detach()
                x0_o_f = x0_o.detach()
                x0_c_f = x0_c.detach()

                target_x1_v = verb_text_features_norm[verb_labels]
                target_x1_o = obj_text_features_norm[obj_labels]

                # --- 辅路 A：流匹配 MSE 轨迹训练 ---
                t = torch.rand(B, 1, device=device)
                xt_v = (1 - t) * x0_v_f + t * target_x1_v
                xt_o = (1 - t) * x0_o_f + t * target_x1_o

                pred_v_v_t = self.v_flow(xt_v, t)
                pred_v_o_t = self.o_flow(xt_o, t)

                xt_v_leak = (1 - t) * x0_o_f + t * target_x1_v
                xt_o_leak = (1 - t) * x0_v_f + t * target_x1_o

                pred_v_v_leak_t = self.v_flow(xt_v_leak, t)
                pred_v_o_leak_t = self.o_flow(xt_o_leak, t)

                # --- 辅路 B：FlowComposer 辅助打分 ---
                t_zero = torch.zeros(B, 1, device=device)
                pred_v_v_0 = self.v_flow(x0_v_f, t_zero)
                pred_v_o_0 = self.o_flow(x0_o_f, t_zero)

                norm_v_v_0 = F.normalize(pred_v_v_0, dim=-1)
                norm_v_o_0 = F.normalize(pred_v_o_0, dim=-1)
                pred_a, pred_b = self.composer(norm_v_v_0, norm_v_o_0)

                pred_v_c_0 = pred_a * norm_v_v_0 + pred_b * norm_v_o_0
                pred_x1_c_0 = x0_c_f + 1.0 * pred_v_c_0

                logits_c = None
                logits_c_flow = None # 单独存放 Flow 的得分

                if pairs is not None:
                    train_v_inds, train_o_inds = pairs[:, 0], pairs[:, 1]

                    c2c_graph_logits = p_pair_o[:, train_v_inds, train_o_inds] + p_pair_v[:, train_v_inds, train_o_inds]

                    pair_verb_text = verb_text_features_norm[train_v_inds]
                    pair_obj_text = obj_text_features_norm[train_o_inds]
                    train_pair_text_features = F.normalize(pair_verb_text + pair_obj_text, dim=-1)

                    pred_x1_c_norm = F.normalize(pred_x1_c_0, dim=-1)
                    flow_explicit_logits = pred_x1_c_norm @ train_pair_text_features.t()
                    flow_explicit_logits = flow_explicit_logits * 0.5 + 0.5

                    # 【彻底分离】：将 C2C 地基分数和 Flow 对比分数单独输出，互不干扰！
                    logits_c = c2c_graph_logits
                    logits_c_flow = flow_explicit_logits

                return {
                    "logits_v": logits_v_base,
                    "logits_o": logits_o_base,
                    "logits_c": logits_c,
                    "logits_c_flow": logits_c_flow, # 独立返回供训练计算 Loss
                    
                    "pred_v_v": pred_v_v_t, "pred_v_o": pred_v_o_t,
                    "true_v_v": target_x1_v - x0_v_f, "true_v_o": target_x1_o - x0_o_f,
                    "pred_v_v_leak": pred_v_v_leak_t, "pred_v_o_leak": pred_v_o_leak_t,
                    "true_v_v_leak": target_x1_v - x0_o_f, "true_v_o_leak": target_x1_o - x0_v_f,
                    "pred_a": pred_a, "pred_b": pred_b,
                    "norm_v_v": norm_v_v_0, "norm_v_o": norm_v_o_0,
                    "true_v_c": F.normalize(target_x1_v + target_x1_o, dim=-1) - x0_c_f,
                    "logit_scale": self.logit_scale
                }

            else:
                # ====== 测试阶段 ======
                t_zero = torch.zeros(B, 1, device=device)

                pred_v_v = self.v_flow(x0_v, t_zero)
                pred_v_o = self.o_flow(x0_o, t_zero)

                norm_v_v = F.normalize(pred_v_v, dim=-1)
                norm_v_o = F.normalize(pred_v_o, dim=-1)
                pred_a, pred_b = self.composer(norm_v_v, norm_v_o)

                pred_v_c = pred_a * norm_v_v + pred_b * norm_v_o
                pred_x1_c_0 = x0_c + 1.0 * pred_v_c

                verb_idx, obj_idx = pairs[:, 0], pairs[:, 1]

                c2c_graph_logits = p_pair_o[:, verb_idx, obj_idx] + p_pair_v[:, verb_idx, obj_idx]

                pair_verb_text = verb_text_features_norm[verb_idx]
                pair_obj_text = obj_text_features_norm[obj_idx]
                pair_text_features = F.normalize(pair_verb_text + pair_obj_text, dim=-1)

                pred_x1_c_norm = F.normalize(pred_x1_c_0, dim=-1)
                flow_explicit_logits = pred_x1_c_norm @ pair_text_features.t()
                flow_explicit_logits = flow_explicit_logits * 0.5 + 0.5

                com_logits = c2c_graph_logits + 0.5 * flow_explicit_logits

                return com_logits

    def condition_module(self, v_feat_c, o_feat_c, v_emb, o_emb, n_o, b, c, n_v):
        v_emb_normed = F.normalize(v_emb, dim=1)
        o_emb_normed = F.normalize(o_emb, dim=1)

        f_v_e_o = self.c2c_f_v_e_o_com(
            torch.cat([v_feat_c.unsqueeze(1).repeat(1, n_o, 1), o_emb.unsqueeze(0).repeat(b, 1, 1)], dim=-1).view(-1, c * 2))
        f_v_e_o_norm = F.normalize(f_v_e_o, dim=-1)
        f_v_e_o_norm = f_v_e_o_norm.view(b, n_o, c)

        f_o_e_v = self.c2c_f_o_e_v_com(
            torch.cat([o_feat_c.unsqueeze(1).repeat(1, n_v, 1), v_emb.unsqueeze(0).repeat(b, 1, 1)], dim=-1).view(-1, c * 2))
        f_o_e_v_norm = F.normalize(f_o_e_v, dim=-1)
        f_o_e_v_norm = f_o_e_v_norm.view(b, n_v, c)

        p_v_con_o = torch.einsum('bnc,mc->bnm', f_v_e_o_norm, v_emb_normed) * 0.5 + 0.5
        p_v_con_o = p_v_con_o.permute(0, 2, 1)
        p_o_con_v = torch.einsum('bnc,mc->bnm', f_o_e_v_norm, o_emb_normed) * 0.5 + 0.5
        return p_v_con_o, p_o_con_v

def load_clip_to_cpu(cfg):
    backbone_name = cfg.backbone
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)
    try:
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    model = clip.build_model(state_dict or model.state_dict())
    return model

def build_model(train_dataset,cfg):
    print(f"Loading CLIP (backbone: {cfg.backbone})")
    clip_model = load_clip_to_cpu(cfg)
    clip_model.float()
    print("Building custom CLIP")
    model = CustomCLIP(cfg, train_dataset, clip_model)
    print("Turning off gradients in both the image and the text encoder")
    for name, param in model.named_parameters():
        param.requires_grad_(False)
        if "prompt_learner" in name:
            if cfg.learn_input_method != 'zero':
                if cfg.learn_input_method == 'coop':
                    if 'prompt_vectors' in name:
                        param.requires_grad_(True)
                elif cfg.learn_input_method == 'csp':
                    if 'obj_embedding' in name or 'verb_embedding' in name or 'comp_embedding' in name:
                        param.requires_grad_(True)
                elif cfg.learn_input_method == 'spm':
                    if 'prompt_vectors' in name or 'obj_embedding' in name or 'verb_embedding' in name or 'comp_embedding' in name:
                        param.requires_grad_(True)
        elif 'video_encoder' in name:
            if 'temporal_embedding' in name or 'ln_post' in name or 'Adapter' in name or 'clip_proj' in name:
                param.requires_grad = True
        elif 'c2c' in name or 'flow' in name or 'composer' in name:
            param.requires_grad = True
    return model