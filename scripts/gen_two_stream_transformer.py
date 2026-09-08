"""Generate complete two-stream Transformer mechanism diagrams for CC-StyTr.

This script generates data-driven visualizations showing:
1. Content Encoder: Global Self-Attention from text token
2. Style Encoder: Global Self-Attention from active texture token
3. Fusion Decoder ×3 layers: Self-Attention → Cross-Attention → Gate Modulation

All outputs are clean PNGs without annotations, saved to paper/figures/flowchart_material/transformer_mechanism/.
"""

import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt

from ccstytr.models.network import CCStyTr
from ccstytr.models.structure import pool_map_to_tokens

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "checkpoints/ablation/residual_gating_spatial_alpha/ccstytr_step_080000.pth"
OUT = ROOT / "paper/figures/flowchart_material/transformer_mechanism"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
state = torch.load(CKPT, map_location=device, weights_only=False)
model_cfg = state.get("model_cfg", {})
model = CCStyTr(**(model_cfg or {})).to(device).eval()
model.load_state_dict(state["model"] if "model" in state else state)
print(f"loaded: {CKPT.name} | smca_mode = {state.get('smca_mode', 'gate')}")

SIZE, GRID = 512, 64

def attn_weights(attn, query, key):
    """Extract attention weights via manual QK^T/sqrt(d) computation."""
    b, nq, d = query.shape
    nk = key.shape[1]
    h, hd = attn.num_heads, attn.head_dim
    q = attn.q_proj(query).view(b, nq, h, hd).transpose(1, 2)
    k = attn.k_proj(key).view(b, nk, h, hd).transpose(1, 2)
    w = (q @ k.transpose(-2, -1)) / math.sqrt(hd)
    return w.softmax(-1)[0]  # (h, nq, nk)

# Load images
content_pil = Image.open(ROOT / "data/eval/poster/5890.png").convert("RGB").resize((SIZE, SIZE))
style_pil = Image.open(ROOT / "data/eval/featured/style_flat/western_painting__met_406721.png").convert("RGB").resize((SIZE, SIZE))

content_np = np.array(content_pil).astype(np.float32) / 255.0
style_np = np.array(style_pil).astype(np.float32) / 255.0
content = torch.from_numpy(content_np).permute(2, 0, 1).unsqueeze(0).to(device)
style = torch.from_numpy(style_np).permute(2, 0, 1).unsqueeze(0).to(device)

OUT.mkdir(parents=True, exist_ok=True)
print("\n--- Generating Two-Stream Transformer Mechanism Diagrams ---")

with torch.no_grad():
    # ================= CONTENT ENCODER =================
    ct0 = model.content_embed(content)
    cape = model.cape(ct0, (GRID, GRID))
    x_ct = ct0 + cape
    
    selfatns_content = []
    for i, layer in enumerate(model.content_encoder.layers):
        if i == len(model.content_encoder.layers) - 1:
            ct_final = x_ct
        if i == len(model.content_encoder.layers) - 1:
            sw = attn_weights(layer.self_attn, layer.norm1(x_ct), layer.norm1(x_ct))
            selfatns_content.append(sw)
        x_ct = layer(x_ct)
    
    # Get salient content token
    edge_map = model.edge(content)
    sob_tok = pool_map_to_tokens(edge_map, (GRID, GRID))[0].reshape(GRID, GRID)
    q_ct_text = int(sob_tok.argmax())
    q_ct_bg = int(sob_tok.argmin())
    
    # ===== Content Encoder Last Layer Self-Attention =====
    self_w_ct = selfatns_content[-1][:, q_ct_text, :].mean(0).reshape(GRID, GRID).cpu().numpy()
    self_w_ct = (self_w_ct / self_w_ct.max()).clip(0, 1)
    overlay_ct = content_pil.convert("RGBA")
    hmap_ct = plt.get_cmap("viridis")(self_w_ct)[:, :, :3]
    ov_img_ct = Image.fromarray((hmap_ct * 255 * 0.5).astype(np.uint8))
    ov_img_ct = ov_img_ct.resize((SIZE, SIZE), Image.Resampling.BILINEAR)
    ov_img_ct.putalpha(int(255 * 0.5))
    Image.alpha_composite(overlay_ct, ov_img_ct).convert("RGB").save(OUT / "T-01_content_encoder_selfattn_from_text.png")
    
    # ================= STYLE ENCODER =================
    ys = model.style_embed(style)
    
    selfatns_style = []
    st_final = None
    for i, layer in enumerate(model.style_encoder.layers):
        if i == len(model.style_encoder.layers) - 1:
            st_final = ys
        if i == len(model.style_encoder.layers) - 1:
            sw = attn_weights(layer.self_attn, layer.norm1(ys), layer.norm1(ys))
            selfatns_style.append(sw)
        ys = layer(ys)
    
    # Get most active style token (texture-rich region)
    style_activations = st_final.norm(dim=-1)
    q_st_active = int(style_activations.argmax())
    grid_q_st = [q_st_active // GRID, q_st_active % GRID]
    
    # ===== Style Encoder Last Layer Self-Attention =====
    self_w_st = selfatns_style[-1][:, q_st_active, :].mean(0).reshape(GRID, GRID).cpu().numpy()
    self_w_st = (self_w_st / self_w_st.max()).clip(0, 1)
    overlay_st = style_pil.convert("RGBA")
    hmap_st = plt.get_cmap("viridis")(self_w_st)[:, :, :3]
    ov_img_st = Image.fromarray((hmap_st * 255 * 0.5).astype(np.uint8))
    ov_img_st = ov_img_st.resize((SIZE, SIZE), Image.Resampling.BILINEAR)
    ov_img_st.putalpha(int(255 * 0.5))
    Image.alpha_composite(overlay_st, ov_img_st).convert("RGB").save(OUT / "T-02_style_encoder_selfattn_from_texture.png")
    
    # ===== Style Decouple =====
    texture_tokens, color_mean, color_std = model.style_decouple(st_final)
    
    # ===== Color Statistics Visualized =====
    mu = color_mean[0, 0].cpu().numpy()
    sd = color_std[0, 0].cpu().numpy()
    bar = np.stack([mu, sd])
    bar = (bar - bar.min()) / (bar.max() - bar.min() + 1e-8)
    bar_img = np.repeat(np.repeat(bar, 32, axis=0), 1, axis=1)
    colored = (plt.get_cmap("viridis")(bar_img)[:, :, :3] * 255).astype(np.uint8)
    Image.fromarray(colored).resize((SIZE, 128), Image.NEAREST).save(OUT / "T-03_color_statistics_mu_sigma.png")
    
    # ===== Texture Tokens PCA =====
    txt_feat = texture_tokens[0]
    _, _, v_txt = torch.pca_lowrank(txt_feat - txt_feat.mean(0, keepdim=True), q=3)
    proj_txt = (txt_feat - txt_feat.mean(0, keepdim=True)) @ v_txt
    proj_txt = proj_txt.reshape(GRID, GRID, 3).cpu().numpy()
    proj_txt = (proj_txt - proj_txt.min(axis=(0, 1))) / (proj_txt.max(axis=(0, 1)) - proj_txt.min(axis=(0, 1)) + 1e-8)
    img_full_txt = Image.fromarray((proj_txt * 255).astype(np.uint8)).resize((SIZE, SIZE), Image.BILINEAR)
    img_full_txt.save(OUT / "T-04_texture_tokens_pca.png")
    
    # ================= FUSION DECODER =================
    gates_all, crossws_all, selfatns_fd = [], [], []
    x_hooked = ct_final
    
    for i, layer in enumerate(model.fusion_decoder.layers):
        # Self-attention on content tokens
        self_w = attn_weights(layer.self_attn, layer.norm1(x_hooked), layer.norm1(x_hooked))
        selfatns_fd.append(self_w)
        
        # Cross-attention: content query → style key
        cross_w = attn_weights(layer.cross_attn, layer.norm2(x_hooked), texture_tokens)
        crossws_all.append(cross_w)
        
        # Gate values
        g_per_token = (1.0 - layer.gamma.abs() * sob_tok.flatten()).clamp(0.0, 1.0)
        gates_all.append(g_per_token.cpu().numpy())
        
        # Forward pass
        x_new = x_hooked + layer.dropout(layer.self_attn(layer.norm1(x_hooked), layer.norm1(x_hooked), layer.norm1(x_hooked)))
        crossout = layer.cross_attn(layer.norm2(x_new), texture_tokens, texture_tokens)
        g_scaled = g_per_token.view(1, GRID*GRID, 1).to(crossout.dtype)
        gated_out = crossout * g_scaled
        x_new = x_new + layer.dropout(gated_out)
        x_new = x_new + layer.dropout(layer.ffn(layer.norm3(x_new)))
        x_hooked = x_new
    
    fused = x_hooked
    
    # ===== FD Layer 1: Self-Attention from text query =====
    self_w_fd1 = selfatns_fd[0][:, q_ct_text, :].mean(0).reshape(GRID, GRID).cpu().numpy()
    self_w_fd1 = (self_w_fd1 / self_w_fd1.max()).clip(0, 1)
    overlay_fd = content_pil.convert("RGBA")
    hmap_fd = plt.get_cmap("viridis")(self_w_fd1)[:, :, :3]
    ov_img_fd = Image.fromarray((hmap_fd * 255 * 0.5).astype(np.uint8))
    ov_img_fd = ov_img_fd.resize((SIZE, SIZE), Image.Resampling.BILINEAR)
    ov_img_fd.putalpha(int(255 * 0.5))
    Image.alpha_composite(overlay_fd, ov_img_fd).convert("RGB").save(OUT / "T-05_fd_layer1_selfattn_from_text.png")
    
    # ===== FD Layer 1: Cross-Attention from text query to style =====
    cross_w_fd1 = crossws_all[0][:, q_ct_text, :].mean(0).reshape(GRID, GRID).cpu().numpy()
    cross_w_fd1 = (cross_w_fd1 / cross_w_fd1.max()).clip(0, 1)
    overlay_fd = style_pil.convert("RGBA")
    hmap_fd = plt.get_cmap("plasma")(cross_w_fd1)[:, :, :3]
    ov_img_fd = Image.fromarray((hmap_fd * 255 * 0.5).astype(np.uint8))
    ov_img_fd = ov_img_fd.resize((SIZE, SIZE), Image.Resampling.BILINEAR)
    ov_img_fd.putalpha(int(255 * 0.5))
    Image.alpha_composite(overlay_fd, ov_img_fd).convert("RGB").save(OUT / "T-06_fd_layer1_crossattn_text_to_style.png")
    
    # ===== FD Layer 1: Gate Map (GyYlGn colormap) =====
    gate_fd1 = gates_all[0].reshape(GRID, GRID)
    cmap = plt.get_cmap("RdYlGn")(gate_fd1)[:, :, :3]
    Image.fromarray((cmap * 255).astype(np.uint8)).save(OUT / "T-07_fd_layer1_gate_map.png")
    
    # ===== FD Layer 2 =====
    self_w_fd2 = selfatns_fd[1][:, q_ct_text, :].mean(0).reshape(GRID, GRID).cpu().numpy()
    self_w_fd2 = (self_w_fd2 / self_w_fd2.max()).clip(0, 1)
    overlay_fd = content_pil.convert("RGBA")
    hmap_fd = plt.get_cmap("viridis")(self_w_fd2)[:, :, :3]
    ov_img_fd = Image.fromarray((hmap_fd * 255 * 0.5).astype(np.uint8))
    ov_img_fd = ov_img_fd.resize((SIZE, SIZE), Image.Resampling.BILINEAR)
    ov_img_fd.putalpha(int(255 * 0.5))
    Image.alpha_composite(overlay_fd, ov_img_fd).convert("RGB").save(OUT / "T-08_fd_layer2_selfattn_from_text.png")
    
    cross_w_fd2 = crossws_all[1][:, q_ct_text, :].mean(0).reshape(GRID, GRID).cpu().numpy()
    cross_w_fd2 = (cross_w_fd2 / cross_w_fd2.max()).clip(0, 1)
    overlay_fd = style_pil.convert("RGBA")
    hmap_fd = plt.get_cmap("plasma")(cross_w_fd2)[:, :, :3]
    ov_img_fd = Image.fromarray((hmap_fd * 255 * 0.5).astype(np.uint8))
    ov_img_fd = ov_img_fd.resize((SIZE, SIZE), Image.Resampling.BILINEAR)
    ov_img_fd.putalpha(int(255 * 0.5))
    Image.alpha_composite(overlay_fd, ov_img_fd).convert("RGB").save(OUT / "T-09_fd_layer2_crossattn_text_to_style.png")
    
    gate_fd2 = gates_all[1].reshape(GRID, GRID)
    cmap = plt.get_cmap("RdYlGn")(gate_fd2)[:, :, :3]
    Image.fromarray((cmap * 255).astype(np.uint8)).save(OUT / "T-10_fd_layer2_gate_map.png")
    
    # ===== FD Layer 3 =====
    self_w_fd3 = selfatns_fd[2][:, q_ct_text, :].mean(0).reshape(GRID, GRID).cpu().numpy()
    self_w_fd3 = (self_w_fd3 / self_w_fd3.max()).clip(0, 1)
    overlay_fd = content_pil.convert("RGBA")
    hmap_fd = plt.get_cmap("viridis")(self_w_fd3)[:, :, :3]
    ov_img_fd = Image.fromarray((hmap_fd * 255 * 0.5).astype(np.uint8))
    ov_img_fd = ov_img_fd.resize((SIZE, SIZE), Image.Resampling.BILINEAR)
    ov_img_fd.putalpha(int(255 * 0.5))
    Image.alpha_composite(overlay_fd, ov_img_fd).convert("RGB").save(OUT / "T-11_fd_layer3_selfattn_from_text.png")
    
    cross_w_fd3 = crossws_all[2][:, q_ct_text, :].mean(0).reshape(GRID, GRID).cpu().numpy()
    cross_w_fd3 = (cross_w_fd3 / cross_w_fd3.max()).clip(0, 1)
    overlay_fd = style_pil.convert("RGBA")
    hmap_fd = plt.get_cmap("plasma")(cross_w_fd3)[:, :, :3]
    ov_img_fd = Image.fromarray((hmap_fd * 255 * 0.5).astype(np.uint8))
    ov_img_fd = ov_img_fd.resize((SIZE, SIZE), Image.Resampling.BILINEAR)
    ov_img_fd.putalpha(int(255 * 0.5))
    Image.alpha_composite(overlay_fd, ov_img_fd).convert("RGB").save(OUT / "T-12_fd_layer3_crossattn_text_to_style.png")
    
    gate_fd3 = gates_all[2].reshape(GRID, GRID)
    cmap = plt.get_cmap("RdYlGn")(gate_fd3)[:, :, :3]
    Image.fromarray((cmap * 255).astype(np.uint8)).save(OUT / "T-13_fd_layer3_gate_map.png")
    
    # ===== Final Output: Fused Tokens PCA =====
    f = fused[0]
    _, _, v_f = torch.pca_lowrank(f - f.mean(0, keepdim=True), q=3)
    proj_f = (f - f.mean(0, keepdim=True)) @ v_f
    proj_f = proj_f.reshape(GRID, GRID, 3).cpu().numpy()
    proj_f = (proj_f - proj_f.min(axis=(0, 1))) / (proj_f.max(axis=(0, 1)) - proj_f.min(axis=(0, 1)) + 1e-8)
    img_full = Image.fromarray((proj_f * 255).astype(np.uint8)).resize((SIZE, SIZE), Image.BILINEAR)
    img_full.save(OUT / "T-14_fused_tokens_pca.png")
    
    # ===== Compare Self-Attention across all modules =====
    self_ct = selfatns_content[-1][:, q_ct_text, :].mean().cpu().numpy()
    self_st = selfatns_style[-1][:, q_st_active, :].mean().cpu().numpy()
    self_fd = np.mean([np.mean(sw[:, q_ct_text, :].cpu().numpy()) for sw in selfatns_fd])
    
    fig, ax = plt.subplots(figsize=(5, 3), dpi=150)
    labels = ['Content\nEncoder', 'Style\nEncoder', 'Fusion\nDecoder']
    heights = [self_ct, self_st, self_fd]
    colors_chart = [plt.get_cmap('viridis')(0.6), plt.get_cmap('magma')(0.5), plt.get_cmap('plasma')(0.5)]
    bars = ax.bar(labels, heights, color=colors_chart)
    ax.set_ylim(0, max(heights)*1.2)
    ax.set_ylabel('Mean Attention Weight')
    ax.spines[:].set_visible(True)
    ax.spines[:].set_visible(True)
    plt.tight_layout()
    plt.savefig(OUT / "T-15_self_attn_comparison.png", bbox_inches='tight', pad_inches=0)
    plt.close()

print("done ->", OUT)
for p in sorted(OUT.glob("*.png")):
    print(" ", p.name)
