import torch
import torch.nn as nn
from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from models.vlm_models.text_learner import get_text_learner
import torch.nn.functional as F
from einops import rearrange
import math

_tokenizer = _Tokenizer()

# ==========================================
# 新增模块：时序流匹配专用网络
# ==========================================

class TimeEmbedding(nn.Module):
    """标准的正弦波时间嵌入，告诉网络当前处于推演的哪个时间步"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        # t shape: [B, 1]
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=t.device, dtype=torch.float32) * -embeddings)
        embeddings = t * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        if self.dim % 2 == 1:
            embeddings = F.pad(embeddings, (0, 1))
        return embeddings # [B, dim]

class TemporalFlowNet(nn.Module):
    """联合序列流网络 (Joint Sequence Flow): 处理 [B, T, D] 的高维流形轨迹"""
    def __init__(self, feature_dim, num_frames=8, num_layers=2, nhead=8):
        super(TemporalFlowNet, self).__init__()
        self.feature_dim = feature_dim
        self.num_frames = num_frames

        self.time_mlp = nn.Sequential(
            TimeEmbedding(feature_dim),
            nn.Linear(feature_dim, feature_dim * 2),
            nn.SiLU(),
            nn.Linear(feature_dim * 2, feature_dim)
        )

        self.pos_embed = nn.Parameter(torch.randn(1, num_frames, feature_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=nhead,
            dim_feedforward=feature_dim * 2,
            batch_first=True,
            activation="gelu",
            norm_first=True
        )
        self.temporal_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.velocity_head = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, feature_dim)
        )

    def forward(self, x_seq, t):
        B, T, D = x_seq.shape
        t_emb = self.time_mlp(t) # [B, D]
        t_emb = t_emb.unsqueeze(1).expand(-1, T, -1) # [B, T, D]

        x_input = x_seq + t_emb + self.pos_embed[:, :T, :]
        hidden_states = self.temporal_transformer(x_input) # [B, T, D]
        velocities = self.velocity_head(hidden_states) # [B, T, D]
        return velocities

class TemporalAttention(nn.Module):
    """时序注意力：用于在推演结束后，提炼核心动力源"""
    def __init__(self, feature_dim):
        super(TemporalAttention, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.LayerNorm(feature_dim // 2),
            nn.GELU(),
            nn.Linear(feature_dim // 2, 1)
        )
    def forward(self, x):
        return self.net(x) # [B, T, 1]

# 🔥 创新点 1: Point-to-Set 物品流网络
class PointToSetFlowMLP(nn.Module):
    """专门为静态物品设计的点到集合 (Point-to-Set) 流网络"""
    def __init__(self, feature_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim + 1, feature_dim * 2),
            nn.LayerNorm(feature_dim * 2),
            nn.GELU(),
            nn.Linear(feature_dim * 2, feature_dim)
        )
    def forward(self, x_seq, t):
        # x_seq 此时已经是被广播后的序列了: [B, T, D]
        B, T, D = x_seq.shape
        t_expand = t.unsqueeze(1).expand(-1, T, 1) # 广播 t: [B, T, 1]
        x_t = torch.cat([x_seq, t_expand], dim=-1) # [B, T, D+1]
        return self.net(x_t) # [B, T, D]

# ==========================================
# 原有模块保留
# ==========================================
class FlowComposer(nn.Module):
    def __init__(self, feature_dim):
        super(FlowComposer, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, 2)
        )
    def forward(self, delta_v, delta_o):
        x = torch.cat([delta_v, delta_o], dim=-1)
        coeffs = self.net(x)
        return coeffs[:, 0:1], coeffs[:, 1:2]

class MLP(nn.Module):
    def __init__(self, inp_dim, out_dim, num_layers=1, relu=True, bias=True, dropout=False, norm=False, layers=[]):
        super(MLP, self).__init__()
        mod = []
        incoming = inp_dim
        for layer_ind in range(num_layers - 1):
            if len(layers) == 0: outgoing = incoming
            else: outgoing = layers[layer_ind]
            mod.append(nn.Linear(incoming, outgoing, bias=bias))
            incoming = outgoing
            if norm: mod.append(nn.LayerNorm(outgoing))
            mod.append(nn.ReLU(inplace=True))
            if dropout: mod.append(nn.Dropout(p=0.5))
        mod.append(nn.Linear(incoming, out_dim, bias=bias))
        if relu: mod.append(nn.ReLU(inplace=True))
        self.mod = nn.Sequential(*mod)
    def forward(self, x):
        return self.mod(x)

class MLP_ST(nn.Module):
    def __init__(self, inp_dim, out_dim, num_layers=1, relu=True, bias=True, dropout=False, norm=False, layers=[]):
        super(MLP_ST, self).__init__()
        mod = []
        incoming = inp_dim
        for layer_ind in range(num_layers - 1):
            if len(layers) == 0: outgoing = incoming
            else: outgoing = layers[layer_ind]
            mod.append(nn.Conv1d(incoming, outgoing, kernel_size=3, bias=bias, padding=1))
            incoming = outgoing
            if norm: mod.append(nn.LayerNorm(outgoing))
            mod.append(nn.ReLU(inplace=True))
            if dropout: mod.append(nn.Dropout(p=0.5))
        mod.append(nn.Conv1d(incoming, out_dim, kernel_size=3, bias=bias, padding=1))
        if relu: mod.append(nn.ReLU(inplace=True))
        self.mod = nn.Sequential(*mod)
    def forward(self, x):
        for o in self.mod:
            if isinstance(o, nn.LayerNorm):
                x = x.transpose(1, 2)
                x = o(x)
                x = x.transpose(1, 2)
            else: x = o(x)
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
        self.num_frames = cfg.num_frames
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

        try: fc_emb = cfg.fc_emb.split(',')
        except: fc_emb = [cfg.fc_emb]
        layers = [int(a) for a in fc_emb]

        self.c2c_OE1 = MLP(cfg.feat_dim, int(cfg.emb_dim), relu=cfg.relu, num_layers=cfg.nlayers, norm=True, layers=layers)
        self.c2c_OE2 = MLP(cfg.feat_dim, int(cfg.emb_dim), relu=cfg.relu, num_layers=cfg.nlayers, norm=True, layers=layers)
        self.c2c_VE1 = MLP_ST(cfg.feat_dim, int(cfg.emb_dim), relu=cfg.relu, num_layers=cfg.nlayers, norm=True, layers=layers)
        self.c2c_VE2 = MLP_ST(cfg.feat_dim, int(cfg.emb_dim), relu=cfg.relu, num_layers=cfg.nlayers, norm=True, layers=layers)

        self.c2c_text_v = nn.Linear(cfg.feat_dim, cfg.emb_dim, bias=True)
        self.c2c_text_o = nn.Linear(cfg.feat_dim, cfg.emb_dim, bias=True)
        self.c2c_f_v_e_o_com = nn.Linear(2 * cfg.emb_dim, cfg.emb_dim, bias=True)
        self.c2c_f_o_e_v_com = nn.Linear(2 * cfg.emb_dim, cfg.emb_dim, bias=True)

        self.use_flow = getattr(cfg, 'use_flow', False)
        if self.use_flow:
            self.flow_v_proj = nn.Linear(cfg.emb_dim, cfg.feat_dim)
            self.flow_o_proj = nn.Linear(cfg.emb_dim, cfg.feat_dim)
            self.flow_c_proj = nn.Linear(cfg.feat_dim, cfg.feat_dim)

            # 🔥 创新点 2: 动词高维序列流 (Set-to-Set) + 物品点阵广播流 (Point-to-Set)
            self.v_flow = TemporalFlowNet(cfg.feat_dim, num_frames=cfg.num_frames) 
            self.o_flow = PointToSetFlowMLP(cfg.feat_dim) 
            self.composer = FlowComposer(cfg.feat_dim)
            self.flow_temporal_attn = TemporalAttention(cfg.feat_dim)

            # 🔥 加载离线生成的 8 步物理轨迹特征库作为不更新的先验知识
            # [Num_Verbs, Num_Objects, 8, 512]
            exact_feats = torch.load('./exact_8step_features.pt')
            self.register_buffer('exact_8step_traj', exact_feats)

    def forward(self, video, pairs=None, verb_labels=None, obj_labels=None):
        verb_prompts = self.verb_prompt_learner()
        raw_verb_text_features = self.text_encoder(verb_prompts, self.verb_tokenized_prompts)
        obj_prompts = self.obj_prompt_learner()
        raw_obj_text_features = self.text_encoder(obj_prompts, self.obj_tokenized_prompts)

        verb_text_features = self.c2c_text_v(raw_verb_text_features)
        obj_text_features = self.c2c_text_o(raw_obj_text_features)

        # 1. 提取 512 维原始视频序列特征
        video_features = self.video_encoder(video) # [B, 512, 8]
        vid_feat_raw = video_features.mean(dim=-1) # [B, 512]

        v_feat_seq_512 = video_features.permute(0, 2, 1) # [B, 8, 512]
        T_frames = v_feat_seq_512.shape[1]

        if self.use_flow:
            attn_logits = self.flow_temporal_attn(v_feat_seq_512) # [B, 8, 1]
            attn_weights = torch.softmax(attn_logits, dim=1)      # [B, 8, 1]

        # 3. 产生 C2C 需要的 300 维特征
        o_feat = self.c2c_OE1(vid_feat_raw)        # [B, 300]
        v_feat_t = self.c2c_VE1(video_features)    # [B, 300, 8]
        v_feat_seq_300 = v_feat_t.permute(0, 2, 1) # [B, 8, 300]

        if self.use_flow:
            v_feat = torch.sum(attn_weights * v_feat_seq_300, dim=1) # [B, 300]
        else:
            v_feat = v_feat_t.mean(dim=-1)

        # ==========================================
        # 🔥 原生 Baseline (Vanilla C2C) 逻辑完整恢复
        # ==========================================
        if not self.use_flow:
            o_feat_normed = F.normalize(o_feat, dim=1)
            v_feat_normed = F.normalize(v_feat, dim=1)
            verb_text_features_norm = F.normalize(verb_text_features, dim=-1)
            obj_text_features_norm = F.normalize(obj_text_features, dim=-1)
            
            verb_logits = v_feat_normed @ verb_text_features_norm.t() * 0.5 + 0.5
            obj_logits = o_feat_normed @ obj_text_features_norm.t() * 0.5 + 0.5
            
            b, c = video_features.shape[0], verb_text_features.shape[-1]
            n_v, n_o = verb_logits.shape[-1], obj_logits.shape[-1]

            o_feat_c = self.c2c_OE2(vid_feat_raw)
            v_feat_c = self.c2c_VE2(video_features).mean(dim=-1)

            p_v_con_o, p_o_con_v = self.condition_module(v_feat_c, o_feat_c, verb_text_features, obj_text_features, n_o, b, c, n_v)
            p_pair_o = p_v_con_o * obj_logits.unsqueeze(1)
            p_pair_v = p_o_con_v * verb_logits.unsqueeze(-1)
            
            if self.training:
                return verb_logits, obj_logits, p_pair_v, p_pair_o, video_features, o_feat, v_feat, p_v_con_o, p_o_con_v
            else:
                verb_idx, obj_idx = pairs[:, 0], pairs[:, 1]
                return p_pair_o[:, verb_idx, obj_idx] + p_pair_v[:, verb_idx, obj_idx]

        # ==========================================
        # 🔥 全新时序流匹配架构逻辑
        # ==========================================
        else:
            B = v_feat.shape[0]
            device = video.device

            x0_v_c2c = F.normalize(v_feat, dim=-1)
            x0_o_c2c = F.normalize(o_feat, dim=-1)
            verb_text_norm_c2c = F.normalize(verb_text_features, dim=-1)
            obj_text_norm_c2c = F.normalize(obj_text_features, dim=-1)

            logits_v_base = x0_v_c2c @ verb_text_norm_c2c.t() * 0.5 + 0.5
            logits_o_base = x0_o_c2c @ obj_text_norm_c2c.t() * 0.5 + 0.5

            o_feat_c = self.c2c_OE2(vid_feat_raw)
            v_feat_c = self.c2c_VE2(video_features).mean(dim=-1)
            p_v_con_o, p_o_con_v = self.condition_module(v_feat_c, o_feat_c, verb_text_features, obj_text_features, logits_o_base.shape[-1], B, verb_text_features.shape[-1], logits_v_base.shape[-1])
            p_pair_o = p_v_con_o * logits_o_base.unsqueeze(1)
            p_pair_v = p_o_con_v * logits_v_base.unsqueeze(-1)

            if self.training:
                v_feat_seq_d = v_feat_seq_300.detach()
                o_feat_d = o_feat.detach()
                vid_feat_d = vid_feat_raw.detach()

                # 视觉特征序列映射到 512维 物理空间
                x0_v_flow_seq = F.normalize(self.flow_v_proj(v_feat_seq_d), dim=-1) # [B, 8, 512]
                x0_o_flow = F.normalize(self.flow_o_proj(o_feat_d), dim=-1)         # [B, 512]
                x0_c_flow = F.normalize(self.flow_c_proj(vid_feat_d), dim=-1)       # [B, 512]

                # 🔥 终点锁定：抽取真实组合的 8步 EXACT 文本轨迹序列 [B, 8, 512]
                target_x1_seq = self.exact_8step_traj[verb_labels, obj_labels].to(device)
                target_x1_seq = F.normalize(target_x1_seq, dim=-1)

                t = torch.rand(B, 1, device=device)
                t_seq = t.unsqueeze(1).unsqueeze(2) # [B, 1, 1] 广播给时间步和特征维

                # 🔥 动词：集合到集合 (Set-to-Set 插值)
                xt_v_seq = (1 - t_seq) * x0_v_flow_seq + t_seq * target_x1_seq
                
                # 🔥 物品：点到集合 (Point-to-Set 广播并插值)
                x0_o_flow_broadcast = x0_o_flow.unsqueeze(1).expand(-1, T_frames, -1) # [B, 8, 512]
                xt_o_seq = (1 - t_seq) * x0_o_flow_broadcast + t_seq * target_x1_seq

                # 提取预测的速度场
                pred_v_v_t_seq = self.v_flow(xt_v_seq, t) # [B, 8, 512]
                pred_v_o_t_seq = self.o_flow(xt_o_seq, t) # [B, 8, 512]

                # 为了生成最终的组合权重 a, b，我们先用注意力将其坍缩以配合 Composer
                pred_v_v_t_point = torch.sum(attn_weights * pred_v_v_t_seq, dim=1) # [B, 512]
                pred_v_o_t_point = torch.sum(attn_weights * pred_v_o_t_seq, dim=1) # [B, 512]
                pred_a, pred_b = self.composer(pred_v_v_t_point, pred_v_o_t_point)

                # 🔥 计算真实的物理演化速度轨迹，提供 MSE Loss 监督
                true_v_v_seq = target_x1_seq - x0_v_flow_seq.detach()
                true_v_o_seq = target_x1_seq - x0_o_flow_broadcast.detach()

                # 端到端组合出来的流
                pred_v_c_seq = pred_a.unsqueeze(2) * pred_v_v_t_seq + pred_b.unsqueeze(2) * pred_v_o_t_seq
                x0_c_flow_seq = x0_c_flow.unsqueeze(1).expand(-1, T_frames, -1)
                true_v_c_seq = target_x1_seq - x0_c_flow_seq.detach()

                return {
                    "logits_v": logits_v_base, "logits_o": logits_o_base, "logits_c": None,
                    "pred_v_v_seq": pred_v_v_t_seq,  # 返回给 Loss 用
                    "true_v_v_seq": true_v_v_seq,    # 返回给 Loss 用
                    "pred_v_o_seq": pred_v_o_t_seq,  # 返回给 Loss 用
                    "true_v_o_seq": true_v_o_seq,    # 返回给 Loss 用
                    "pred_v_c_seq": pred_v_c_seq,    # 返回给 Loss 用
                    "true_v_c_seq": true_v_c_seq,    # 返回给 Loss 用
                    "pred_a": pred_a, "pred_b": pred_b,
                    "logit_scale": self.logit_scale
                }

            else:
                t_zero = torch.zeros(B, 1, device=device)

                x0_v_flow_seq = F.normalize(self.flow_v_proj(v_feat_seq_300), dim=-1) # [B, 8, 512]
                x0_o_flow = F.normalize(self.flow_o_proj(o_feat), dim=-1)             # [B, 512]
                x0_c_flow = F.normalize(self.flow_c_proj(vid_feat_raw), dim=-1)       # [B, 512]

                x0_o_flow_broadcast = x0_o_flow.unsqueeze(1).expand(-1, T_frames, -1) # [B, 8, 512]

                # 自由推演 8步 速度序列
                pred_v_v_0_seq = self.v_flow(x0_v_flow_seq, t_zero) # [B, 8, 512]
                pred_v_o_0_seq = self.o_flow(x0_o_flow_broadcast, t_zero) # [B, 8, 512]

                # 计算融合权重
                delta_v_0_point = torch.sum(attn_weights * pred_v_v_0_seq, dim=1) 
                delta_o_0_point = torch.sum(attn_weights * pred_v_o_0_seq, dim=1) 
                pred_a, pred_b = self.composer(delta_v_0_point, delta_o_0_point)

                # 融合生成最终想象的 8步 组合预测轨迹
                pred_v_c_seq = pred_a.unsqueeze(2) * pred_v_v_0_seq + pred_b.unsqueeze(2) * pred_v_o_0_seq
                
                # 基准状态广播为序列
                x0_c_flow_seq = x0_c_flow.unsqueeze(1).expand(-1, T_frames, -1)
                pred_x1_c_seq = x0_c_flow_seq + 1.0 * pred_v_c_seq # [B, 8, 512]

                verb_idx, obj_idx = pairs[:, 0], pairs[:, 1]
                c2c_graph_logits = p_pair_o[:, verb_idx, obj_idx] + p_pair_v[:, verb_idx, obj_idx]

                # 🔥 测试期：Trajectory-to-Trajectory 全局物理路径检索
                # 提取候选库中的标准 8 步物理文本路径
                test_target_seqs = self.exact_8step_traj[verb_idx, obj_idx].to(device) # [P, 8, 512]
                
                pred_traj_norm = F.normalize(pred_x1_c_seq, dim=-1) # [B, 8, 512]
                target_traj_norm = F.normalize(test_target_seqs, dim=-1) # [P, 8, 512]

                # 计算 8 步的余弦相似度，并沿时间维度求均值
                sim_traj = torch.einsum('btd,ptd->bpt', pred_traj_norm, target_traj_norm) # [B, P, 8]
                flow_explicit_logits = sim_traj.mean(dim=-1) * 0.5 + 0.5 # [B, P]

                # 与基座的 logits 融合
                com_logits = c2c_graph_logits + 1.0 * flow_explicit_logits
                return com_logits

    def condition_module(self, v_feat_c, o_feat_c, v_emb, o_emb, n_o, b, c, n_v):
        v_emb_normed = F.normalize(v_emb, dim=1)
        o_emb_normed = F.normalize(o_emb, dim=1)
        f_v_e_o = self.c2c_f_v_e_o_com(torch.cat([v_feat_c.unsqueeze(1).repeat(1, n_o, 1), o_emb.unsqueeze(0).repeat(b, 1, 1)], dim=-1).view(-1, c * 2))
        f_v_e_o_norm = F.normalize(f_v_e_o, dim=-1).view(b, n_o, c)
        f_o_e_v = self.c2c_f_o_e_v_com(torch.cat([o_feat_c.unsqueeze(1).repeat(1, n_v, 1), v_emb.unsqueeze(0).repeat(b, 1, 1)], dim=-1).view(-1, c * 2))
        f_o_e_v_norm = F.normalize(f_o_e_v, dim=-1).view(b, n_v, c)
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
                if cfg.learn_input_method in ['coop', 'csp', 'spm']:
                    if any(key in name for key in ['prompt_vectors', 'obj_embedding', 'verb_embedding', 'comp_embedding']):
                        param.requires_grad_(True)
        elif 'video_encoder' in name:
            if any(key in name for key in ['temporal_embedding', 'ln_post', 'Adapter', 'clip_proj']):
                param.requires_grad = True
        elif any(key in name for key in ['c2c', 'flow', 'composer']):
            param.requires_grad = True
    return model