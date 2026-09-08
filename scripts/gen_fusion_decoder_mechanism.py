"""Generate simplified Fusion Decoder mechanism diagrams (core attention patterns only).

Focus on: self-attention, cross-attention, gate magnitude comparison.
All outputs are clean PNGs without text/annotations.
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
OUT = ROOT / "paper/figures/flowchart_material/fusion_decoder_mechanism"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
state = torch.load(CKPT, map_location=device, weights_only=False)
model_cfg = state.get("model_cfg", {})
model = CCStyTr(**(model_cfg or {})).to(device).eval()
model.load_state_dict(state["model"] if "model" in state else state)
print(f"loaded: {CKPT.name} | smca_mode = {state.get('smca_mode', 'gate')}")

SIZE, GRID = 512, 64

def attn_weights(attn, query, key):
    b, nq, d = query.shape
    nk = key.shape[1]
    h, hd = attn.num_heads, attn.head_dim
    q = attn.q_proj(query).view(b, nq, h, hd).transpose(1, 2)
    k = attn.k_proj(key).view(b, nk, h, hd).transpose(1, 2)
    w = (q @ k.transpose(-2, -1)) / math.sqrt(hd)
    return w.softmax(-1)[0]

# Load images
content_pil = Image.open(ROOT / "data/eval/poster/5890.png").convert("RGB").resize((SIZE, SIZE))
style_pil = Image.open(ROOT / "data/eval/featured/style_flat/western_painting__met_406721.png").convert("RGB").resize((SIZE, SIZE))

content_np = np.array(content_pil).astype(np.float32) / 255.0
style_np = np.array(style_pil).astype(np.float32) / 255.0
content = torch.from_numpy(content_np).permute(2, 0, 1).unsqueeze(0).to(device)
style = torch.from_numpy(style_np).permute(2, 0, 1).unsqueeze(0).to(device)

OUT.mkdir(parents=True, exist_ok=True)
print("\n--- Generating Fusion Decoder Mechanism Diagrams ---")

with torch.no_grad():
    # Encode
    ct0 = model.content_embed(content)
    cape = model.cape(ct0, (GRID, GRID))
    x = ct0 + cape
    for i, layer in enumerate(model.content_encoder.layers):
        if i == len(model.content_encoder.layers) - 1:
            ct_final = x
        x = layer(x)
    
    ys = model.style_embed(style)
    st_final = None
    for i, layer in enumerate(model.style_encoder.layers):
        if i == len(model.style_encoder.layers) - 1:
            st_final = ys
        ys = layer(ys)
    
    texture_tokens, color_mean, color_std = model.style_decouple(st_final)
    
    # Get saliency
    edge_map = model.edge(content)
    sob_tok = pool_map_to_tokens(edge_map, (GRID, GRID))[0]  # (N,)
    q_text = int(sob_tok.argmax())
    q_bg = int(sob_tok.argmin())
    
    gates_vals = []
    crossws_all, selfatns_all = [], []
    
    x_hooked = ct_final
    for i, layer in enumerate(model.fusion_decoder.layers):
        # Self-attention
        self_w = attn_weights(layer.self_attn, layer.norm1(x_hooked), layer.norm1(x_hooked))
        selfatns_all.append(self_w)
        
        # Cross-attention
        cross_w = attn_weights(layer.cross_attn, layer.norm2(x_hooked), texture_tokens)
        crossws_all.append(cross_w)
        
        # Gate values per token
        g_per_token = (1.0 - layer.gamma.abs() * sob_tok).clamp(0.0, 1.0)
        gates_vals.append(g_per_token.cpu().numpy())
        
        # Forward pass
        x_new = x_hooked + layer.dropout(layer.self_attn(layer.norm1(x_hooked), layer.norm1(x_hooked), layer.norm1(x_hooked)))
        crossout = layer.cross_attn(layer.norm2(x_new), texture_tokens, texture_tokens)
        g_scaled = g_per_token.view(1, GRID*GRID, 1).to(crossout.dtype)
        gated_out = crossout * g_scaled
        x_new = x_new + layer.dropout(gated_out)
        x_new = x_new + layer.dropout(layer.ffn(layer.norm3(x_new)))
        x_hooked = x_new
    
    fused = x_hooked
    
    # Generate visualizations
    
    # M-01: Self-attention from salient text query (last layer)
    amap_self = selfatns_all[-1][:, q_text, :].mean(0).reshape(GRID, GRID).cpu().numpy()
    amap_self = (amap_self / amap_self.max()).clip(0, 1)
    overlay = content_pil.convert("RGBA")
    hmap = plt.get_cmap("magma")(amap_self)[:, :, :3]  # RGB only
    ov_img = Image.fromarray((hmap * 255 * 0.5).astype(np.uint8))
    ov_img = ov_img.resize((SIZE, SIZE), Image.Resampling.BILINEAR)  # Resize to full size
    ov_img.putalpha(int(255 * 0.5))
    Image.alpha_composite(overlay, ov_img).convert("RGB").save(OUT / "M-01_self_attn_from_text.png")
    
    # M-02: Cross-attention from salient text query to style tokens (last layer)
    amap_cross_text = crossws_all[-1][:, q_text, :].mean(0).reshape(GRID, GRID).cpu().numpy()
    amap_cross_text = (amap_cross_text / amap_cross_text.max()).clip(0, 1)
    overlay = style_pil.convert("RGBA")
    hmap = plt.get_cmap("plasma")(amap_cross_text)[:, :, :3]  # RGB only
    ov_img = Image.fromarray((hmap * 255 * 0.5).astype(np.uint8))
    ov_img = ov_img.resize((SIZE, SIZE), Image.Resampling.BILINEAR)  # Resize to full size
    ov_img.putalpha(int(255 * 0.5))
    Image.alpha_composite(overlay, ov_img).convert("RGB").save(OUT / "M-02_cross_attn_text_to_style.png")
    
    # M-03: Cross-attention from background query to style tokens
    amap_cross_bg = crossws_all[-1][:, q_bg, :].mean(0).reshape(GRID, GRID).cpu().numpy()
    amap_cross_bg = (amap_cross_bg / amap_cross_bg.max()).clip(0, 1)
    overlay = style_pil.convert("RGBA")
    hmap = plt.get_cmap("plasma")(amap_cross_bg)[:, :, :3]  # RGB only
    ov_img = Image.fromarray((hmap * 255 * 0.5).astype(np.uint8))
    ov_img = ov_img.resize((SIZE, SIZE), Image.Resampling.BILINEAR)  # Resize to full size
    ov_img.putalpha(int(255 * 0.5))
    Image.alpha_composite(overlay, ov_img).convert("RGB").save(OUT / "M-03_cross_attn_bg_to_style.png")
    
    # M-04~06: Gate maps per layer (RdYlGn colormap)
    gates_grid_all = []
    for i, g_per_tok in enumerate(gates_vals):
        g_grid = g_per_tok.reshape(GRID, GRID)
        gates_grid_all.append(g_grid)
        cmap = plt.get_cmap("RdYlGn")(g_grid)[:, :, :3]
        Image.fromarray((cmap * 255).astype(np.uint8)).save(OUT / f"M-0{4+i}_gate_layer{i+1}.png")
    
    # M-07: Fused tokens PCA projection (3rd output channel visualization)
    f = fused[0]
    _, _, v_f = torch.pca_lowrank(f - f.mean(0, keepdim=True), q=3)
    proj_f = (f - f.mean(0, keepdim=True)) @ v_f
    proj_f = proj_f.reshape(GRID, GRID, 3).cpu().numpy()
    proj_f = (proj_f - proj_f.min(axis=(0, 1))) / (proj_f.max(axis=(0, 1)) - proj_f.min(axis=(0, 1)) + 1e-8)
    img_full = Image.fromarray((proj_f * 255).astype(np.uint8)).resize((SIZE, SIZE), Image.BILINEAR)
    img_full.save(OUT / "M-07_fused_tokens_pca.png")

print("done ->", OUT)
for p in sorted(OUT.glob("*.png")):
    print(" ", p.name)
