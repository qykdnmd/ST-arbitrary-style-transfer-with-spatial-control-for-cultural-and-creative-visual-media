"""Generate complete flowchart materials for CC-StyTr two-stream architecture.

This script generates visualizations organized by flow stage:
- Stage C (Content Stream): Content image → PatchEmbed → CAPE → Encoder → Output
- Stage S (Style Stream): Style image → PatchEmbed → Style Encoder → Decouple → Texture/Color tokens
- Fusion: Gate mechanism + Cross-attention → Alpha-map → CNN Decoder → Final output

Example pair: RSNY MEN poster (content) x Fiery-tailed Sun Bird (style).
All outputs are clean PNGs (512x512) without text annotations, saved to paper/figures/flowchart_material/.
"""

import math
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw

from ccstytr.data.datasets import test_transform
from ccstytr.models.network import CCStyTr
from ccstytr.models.structure import pool_map_to_tokens

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "checkpoints/ablation/residual_gating_spatial_alpha/ccstytr_step_080000.pth"
OUT = ROOT / "paper/figures/flowchart_material"

# Load checkpoint and model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
state = torch.load(CKPT, map_location=device, weights_only=False)
model_cfg = state.get("model_cfg", {})
model = CCStyTr(**(model_cfg or {})).to(device).eval()
model.load_state_dict(state["model"] if "model" in state else state)
smca_mode = state.get("smca_mode", "gate")
print(f"loaded: {CKPT.name} | smca_mode = {smca_mode}")

# Example pairs
CONTENT = ROOT / "data/eval/poster/5890.png"        # RSNY MEN ring poster (English text)
STYLE = ROOT / "data/eval/featured/style_flat/western_painting__met_406721.png"  # Fiery-tailed Sun Bird

SIZE = 512
GRID = 64
PATCH = 8

def overlay_heatmap(base_img: Image.Image, heatmap: np.ndarray, alpha: float = 0.5, cmap: str = "Greens"):
    """Overlay a normalized heatmap (0-1) on base image using given colormap."""
    base = base_img.convert("RGBA")
    hmap = (heatmap * 255).astype(np.uint8)
    hmap_colored = plt.get_cmap(cmap)(hmap)[:, :, :3]
    ov = Image.fromarray((hmap_colored * 255 * alpha).astype(np.uint8))
    ov.putalpha(int(255 * (1 - alpha)))
    # Resize heatmap to match base image
    if ov.size != base.size:
        ov = ov.resize(base.size, Image.Resampling.BILINEAR)
    return Image.alpha_composite(base, ov).convert("RGB")

def save_heatmap_only(hmap: np.ndarray, filename: str, cmap: str = "Greens"):
    """Save heatmap as pure grayscale/colormap visualization."""
    hmap = (hmap * 255).astype(np.uint8)
    hmap_colored = plt.get_cmap(cmap)(hmap)[:, :, :3]  # RGB only
    Image.fromarray(hmap_colored.astype(np.uint8)).save(OUT / filename)

def attn_weights(attn, query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
    """Manual softmax(QK^T/sqrt(d)) to extract attention weights."""
    b, nq, d = query.shape
    nk = key.shape[1]
    h, hd = attn.num_heads, attn.head_dim
    q = attn.q_proj(query).view(b, nq, h, hd).transpose(1, 2)
    k = attn.k_proj(key).view(b, nk, h, hd).transpose(1, 2)
    w = (q @ k.transpose(-2, -1)) / math.sqrt(hd)
    return w.softmax(-1)[0]  # (h, nq, nk)

# Ensure output directory exists
OUT.mkdir(parents=True, exist_ok=True)

# Load images
content_pil = Image.open(CONTENT).convert("RGB").resize((SIZE, SIZE), Image.Resampling.LANCZOS)
style_pil = Image.open(STYLE).convert("RGB").resize((SIZE, SIZE), Image.Resampling.LANCZOS)

# Convert PIL to tensor manually (avoid test_transform which is a Compose)
content_np = np.array(content_pil).astype(np.float32) / 255.0
style_np = np.array(style_pil).astype(np.float32) / 255.0
content = torch.from_numpy(content_np).permute(2, 0, 1).unsqueeze(0).to(device)  # (1,3,H,W)
style = torch.from_numpy(style_np).permute(2, 0, 1).unsqueeze(0).to(device)

def draw_grid(img: Image.Image) -> Image.Image:
    """Overlay white grid lines on full image (reference fig 1 style)."""
    out = img.convert("RGBA")
    ov = Image.new("RGBA", out.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    for i in range(1, GRID):
        p = i * PATCH
        d.line([(p, 0), (p, SIZE)], fill=(255, 255, 255, 110))
        d.line([(0, p), (SIZE, p)], fill=(255, 255, 255, 110))
    return Image.alpha_composite(out, ov).convert("RGB")

def draw_separated(img: Image.Image, gap: int = 2) -> Image.Image:
    """Patches physically separated by transparent gaps (W-MSA-window look)."""
    canvas_size = SIZE + (GRID - 1) * gap
    out = Image.new("RGBA", (canvas_size, canvas_size), (0, 0, 0, 0))
    src = img.convert("RGBA")
    for gy in range(GRID):
        for gx in range(GRID):
            tile = src.crop((gx * PATCH, gy * PATCH, (gx + 1) * PATCH, (gy + 1) * PATCH))
            out.paste(tile, (gx * (PATCH + gap), gy * (PATCH + gap)))
    return out


# ==================== STAGE C: CONTENT STREAM ====================
print("\n--- Generating CONTENT STREAM materials ---")

# C-01: Raw content input
content_pil.save(OUT / "C-01_raw_input.png")

# C-02: Grid overlay (patch partition visualization)
draw_grid(content_pil).save(OUT / "C-02_patch_grid.png")

# C-03: Separated patches (window-style attention illustration)
draw_separated(content_pil).save(OUT / "C-03_separated_patches.png")

# C-04: PatchEmbed → Token amplitude (L2 norm compression)
with torch.no_grad():
    ct_embed = model.content_embed(content)
    ct_proj = ct_embed[0].cpu().numpy()
    
    feat = np.linalg.norm(ct_proj, axis=1) / np.sqrt(512)
    feat /= feat.max() + 1e-8
    img_ct = feat.reshape(GRID, GRID)
    Image.fromarray((255.0 * img_ct).astype(np.uint8)).save(OUT / "C-04_token_amplitude.png")
    
    # C-05: Patch embedding schematic (teaching diagram)
    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(8, 2.2), dpi=200)
    src = np.asarray(content_pil.convert("RGB"))
    gx, gy = 5, 3
    patch = src[gy*PATCH:(gy+1)*PATCH, gx*PATCH:(gx+1)*PATCH].astype(np.float32) / 255.0
    ax0.imshow(patch)
    ax0.set_title("patch x ∈ R¹⁹²", fontsize=10, ha="center")
    ax0.axis("off")
    ax1.text(0.5, 0.5, "W ∈ R¹⁹²ˣ⁵¹²\n(learned weights)", fontsize=14, va="center", ha="center")
    ax1.axis("off")
    ax2.text(0.5, 0.5, "h = ReLU(W.T x) ∈ R⁵¹²\n(output token)", fontsize=14, va="center", ha="center")
    ax2.axis("off")
    plt.savefig(OUT / "C-05_patch_schematic.png", bbox_inches="tight", pad_inches=0.1)
    plt.close()
    
    # C-06: RGB feature field via mean/variance
    subfeat = ct_proj[:GRID*GRID][:, :64]
    mean_per_token = subfeat.mean(axis=1, keepdims=True)
    var_per_token = subfeat.var(axis=1, keepdims=True)
    r = (mean_per_token - mean_per_token.min()) / (mean_per_token.max() - mean_per_token.min() + 1e-8)
    g = (var_per_token - var_per_token.min()) / (var_per_token.max() - var_per_token.min() + 1e-8)
    b = np.zeros_like(r); b[:] = 0.5
    img_rgb = np.stack([r, g, b], axis=-1).reshape(GRID, GRID, 3)
    img_full = cv2.resize(img_rgb, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)
    Image.fromarray((img_full * 255).astype(np.uint8)).save(OUT / "C-06_rgb_feature_field.png")
    
    # C-07: CAPE magnitude map
    cape = model.cape(ct_embed, (GRID, GRID))
    cape_mag = cape[0].norm(dim=-1).reshape(GRID, GRID).cpu().numpy()
    cape_mag /= cape_mag.max() + 1e-8
    overlay_heatmap(content_pil, cape_mag, alpha=0.45, cmap="Greens").save(OUT / "C-07_cape_overlay.png")
    
    # Save standalone CAPE map
    cmap = plt.get_cmap("Greens")
    cape_vis = (cmap(cape_mag)[..., :3] * 255).astype(np.uint8)
    Image.fromarray(cape_vis).save(OUT / "C-08_cape_standalone.png")
    
    ct_in = ct_embed + cape
    
    # Self-attention from text token
    x = ct_in
    for i, layer in enumerate(model.content_encoder.layers):
        if i == len(model.content_encoder.layers) - 1:
            amap_w = attn_weights(layer.self_attn, layer.norm1(x), layer.norm1(x))
        x = layer(x)
    ct_final = x
    
    q_text = int(cape_mag.flatten().argmax())  # most salient token
    amap = amap_w[:, q_text, :].mean(0).reshape(GRID, GRID).cpu().numpy()
    amap /= amap.max() + 1e-8
    overlay_heatmap(content_pil, amap, alpha=0.55, cmap="magma").save(OUT / "C-09_selfattn_from_text.png")
    
    # Content encoder final output (PCA projection)
    f = ct_final[0]
    _, _, v = torch.pca_lowrank(f - f.mean(0, keepdim=True), q=3)
    proj = (f - f.mean(0, keepdim=True)) @ v
    proj = proj.reshape(GRID, GRID, 3).cpu().numpy()
    proj = (proj - proj.min(axis=(0, 1))) / (proj.max(axis=(0, 1)) - proj.min(axis=(0, 1)) + 1e-8)
    Image.fromarray((proj * 255).astype(np.uint8)).resize((SIZE, SIZE), Image.BILINEAR).save(OUT / "C-10_final_tokens.png")


# ==================== STAGE S: STYLE STREAM ====================
print("\n--- Generating STYLE STREAM materials ---")

# S-01: Raw style input
style_pil.save(OUT / "S-01_raw_input.png")

# S-02: Grid overlay
draw_grid(style_pil).save(OUT / "S-02_patch_grid.png")

# S-03: Separated patches
draw_separated(style_pil).save(OUT / "S-03_separated_patches.png")

# S-04: PatchEmbed → Token amplitude
with torch.no_grad():
    st_embed = model.style_embed(style)
    st_proj = st_embed[0].cpu().numpy()
    
    feat = np.linalg.norm(st_proj, axis=1) / np.sqrt(512)
    feat /= feat.max() + 1e-8
    img_st = feat.reshape(GRID, GRID)
    Image.fromarray((255.0 * img_st).astype(np.uint8)).save(OUT / "S-04_token_amplitude.png")
    
    # S-05: RGB feature field
    subfeat = st_proj[:GRID*GRID][:, :64]
    mean_per_token = subfeat.mean(axis=1, keepdims=True)
    var_per_token = subfeat.var(axis=1, keepdims=True)
    r = (mean_per_token - mean_per_token.min()) / (mean_per_token.max() - mean_per_token.min() + 1e-8)
    g = (var_per_token - var_per_token.min()) / (var_per_token.max() - var_per_token.min() + 1e-8)
    b = np.zeros_like(r); b[:] = 0.5
    img_rgb = np.stack([r, g, b], axis=-1).reshape(GRID, GRID, 3)
    img_full = cv2.resize(img_rgb, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)
    Image.fromarray((img_full * 255).astype(np.uint8)).save(OUT / "S-06_rgb_feature_field.png")
    
    ys = st_embed
    texture_tokens, color_mean, color_std = None, None, None
    for i, layer in enumerate(model.style_encoder.layers):
        if i == len(model.style_encoder.layers) - 1:
            smap_w = attn_weights(layer.self_attn, layer.norm1(ys), layer.norm1(ys))
        ys = layer(ys)
    st_final = ys
    
    q_style = int(st_embed[0].norm(dim=-1).argmax())
    smap = smap_w[:, q_style, :].mean(0).reshape(GRID, GRID).cpu().numpy()
    smap /= smap.max() + 1e-8
    overlay_heatmap(style_pil, smap, alpha=0.55, cmap="magma").save(OUT / "S-07_selfattn_from_active.png")
    
    # Style decoupling
    texture_tokens, color_mean, color_std = model.style_decouple(st_final)
    
    # S-08: Color statistics (μ/σ bars)
    mu = color_mean[0, 0].cpu().numpy()
    sd = color_std[0, 0].cpu().numpy()
    bar = np.stack([mu, sd])
    bar = (bar - bar.min()) / (bar.max() - bar.min() + 1e-8)
    bar_img = np.repeat(np.repeat(bar, 32, axis=0), 1, axis=1)
    colored = (plt.get_cmap("viridis")(bar_img)[..., :3] * 255).astype(np.uint8)
    Image.fromarray(colored).resize((SIZE, 128), Image.NEAREST).save(OUT / "S-08_color_stats.png")
    
    # Save texture tokens as PCA projection
    txt_feat = texture_tokens[0]
    _, _, v_txt = torch.pca_lowrank(txt_feat - txt_feat.mean(0, keepdim=True), q=3)
    txt_proj = (txt_feat - txt_feat.mean(0, keepdim=True)) @ v_txt
    txt_proj = txt_proj.reshape(GRID, GRID, 3).cpu().numpy()
    txt_proj = (txt_proj - txt_proj.min(axis=(0, 1))) / (txt_proj.max(axis=(0, 1)) - txt_proj.min(axis=(0, 1)) + 1e-8)
    Image.fromarray((txt_proj * 255).astype(np.uint8)).resize((SIZE, SIZE), Image.BILINEAR).save(OUT / "S-09_texture_tokens.png")


# ==================== FUSION & DECODING ====================
print("\n--- Generating FUSION & DECODING materials ---")

with torch.no_grad():
    # Sobel edge map
    edge_map = model.edge(content)
    sob = edge_map[0, 0].cpu().numpy()
    save_heatmap_only(sob, "F-01_sobel_edge_map.png", cmap="gray")
    
    # Salience pooling
    sal_tok = pool_map_to_tokens(edge_map, (GRID, GRID))[0]
    sal_grid = sal_tok.reshape(GRID, GRID).cpu().numpy()
    overlay_heatmap(content_pil, sal_grid, alpha=0.5).save(OUT / "F-02_token_salience.png")
    
    # Gate maps (residual_gating_spatial_alpha configuration)
    gates = []
    crossws = []
    x = ct_final
    for layer in model.fusion_decoder.layers:
        x = x + layer.dropout(layer.self_attn(layer.norm1(x), layer.norm1(x), layer.norm1(x)))
        g = (1.0 - layer.gamma.abs() * sal_tok).clamp(0.0, 1.0)
        gates.append(g.reshape(GRID, GRID).cpu().numpy())
        crossw = attn_weights(layer.cross_attn, layer.norm2(x), texture_tokens)
        crossws.append(crossw)
        attn_out = layer.cross_attn(layer.norm2(x), texture_tokens, texture_tokens)
        x = x + layer.dropout(g.unsqueeze(-1).to(attn_out.dtype).unsqueeze(0) * attn_out)
        x = x + layer.dropout(layer.ffn(layer.norm3(x)))
    fused = x
    
    # Gate layers 1/2/3
    for i, g in enumerate(gates, 1):
        overlay_heatmap(content_pil, g, alpha=0.5, cmap="RdYlGn").save(
            OUT / f"F-0{1+i}_gate_layer{i}.png")
    
    # Cross-attention (text vs bg)
    wl = crossws[-1]
    for name, qi in [("F-05_crossattn_over_text.png", q_text),
                     ("F-06_crossattn_over_bg.png", int(sal_tok.argmin()))]:
        m = wl[:, qi, :].mean(0).reshape(GRID, GRID).cpu().numpy()
        m /= m.max() + 1e-8
        overlay_heatmap(style_pil, m, alpha=0.55, cmap="magma").save(OUT / name)
    
    # Fused tokens PCA
    f = fused[0]
    _, _, vf = torch.pca_lowrank(f - f.mean(0, keepdim=True), q=3)
    proj_f = (f - f.mean(0, keepdim=True)) @ vf
    proj_f = proj_f.reshape(GRID, GRID, 3).cpu().numpy()
    proj_f = (proj_f - proj_f.min(axis=(0, 1))) / (proj_f.max(axis=(0, 1)) - proj_f.min(axis=(0, 1)) + 1e-8)
    Image.fromarray((proj_f * 255).astype(np.uint8)).resize((SIZE, SIZE), Image.BILINEAR).save(OUT / "F-07_fused_tokens.png")
    
    # Demo alpha map
    alpha_demo = (1.0 - 0.7 * torch.from_numpy(sob)).clamp(0.0, 1.0).to(device)
    save_heatmap_only(alpha_demo.cpu().numpy(), "F-08_alpha_map_demo.png", cmap="RdBu_r")
    
    # Outputs at different α levels
    for name, aval in [("F-09_output_α=0.0.png", 0.0),
                       ("F-10_output_α=0.5.png", 0.5),
                       ("F-11_output_α=1.0.png", 1.0)]:
        amap_t = torch.full((1, 1, SIZE, SIZE), aval, device=device)
        out = model(content, style, amap_t)
        # to_pil: normalize from [-1,1] to [0,1]
        def to_pil(t):
            t = t.squeeze(0)  # (C,H,W)
            img = t.cpu().numpy().transpose(1,2,0)  # (H,W,C)
            img = ((img + 1)/2 * 255).clip(0,255).astype(np.uint8)
            return Image.fromarray(img)
        to_pil(out).save(OUT / name)
    
    # Main demo output with spatial alpha
    out_demo = model(content, style, alpha_demo.unsqueeze(0).unsqueeze(0))
    to_pil(out_demo[0].cpu()).save(OUT / "F-12_final_output.png")

print("\ndone ->", OUT)
for p in sorted(OUT.glob("*.png")):
    print(" ", p.name)
