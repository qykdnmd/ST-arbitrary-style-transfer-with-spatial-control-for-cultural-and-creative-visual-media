"""Generate flowchart material images for the CC-StyTr architecture figure.

Example pair: poster 1041 (content) x chinese_landscape__130942 (style).
Model: proposed residual_gating_spatial_alpha 80k checkpoint. Outputs are PNGs (512x512)
without text annotations, saved to paper/figures/flowchart_material/.
"""
import math
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import torch
from PIL import Image, ImageDraw

from ccstytr.data.datasets import test_transform
from ccstytr.models.network import CCStyTr
from ccstytr.models.structure import pool_map_to_tokens

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "checkpoints/ablation/residual_gating_spatial_alpha/ccstytr_step_080000.pth"
CONTENT = ROOT / "data/eval/poster/5890.png"
STYLE = ROOT / "data/eval/featured/style_flat/western_painting__met_406721.png"
OUT = ROOT / "paper/figures/flowchart_material"
SIZE = 512
PATCH = 8
GRID = SIZE // PATCH  # 64


def to_pil(t: torch.Tensor) -> Image.Image:
    return Image.fromarray((t.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8))


def overlay_heatmap(img: Image.Image, heat: np.ndarray, alpha: float = 0.5,
                    cmap: str = "viridis") -> Image.Image:
    """heat: (64,64) or (H,W) array in [0,1]; upsample and blend onto img."""
    h = np.asarray(Image.fromarray((heat * 255).astype(np.uint8))
                   .resize((SIZE, SIZE), Image.BILINEAR), dtype=np.float32) / 255.0
    colored = matplotlib.colormaps[cmap](h)[..., :3].astype(np.float32)
    base = np.asarray(img, dtype=np.float32) / 255.0
    out = (1 - alpha) * base + alpha * colored
    return Image.fromarray((out.clip(0, 1) * 255).astype(np.uint8))


def save_heatmap_only(heat: np.ndarray, name: str, cmap: str = "viridis"):
    h = np.asarray(Image.fromarray((heat * 255).astype(np.uint8))
                   .resize((SIZE, SIZE), Image.BILINEAR), dtype=np.float32) / 255.0
    colored = (matplotlib.colormaps[cmap](h)[..., :3] * 255).astype(np.uint8)
    Image.fromarray(colored).save(OUT / name)


def draw_grid(img: Image.Image) -> Image.Image:
    out = img.convert("RGBA")
    ov = Image.new("RGBA", out.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    for i in range(1, GRID):
        p = i * PATCH
        d.line([(p, 0), (p, SIZE)], fill=(255, 255, 255, 110))
        d.line([(0, p), (SIZE, p)], fill=(255, 255, 255, 110))
    return Image.alpha_composite(out, ov).convert("RGB")


def draw_separated(img: Image.Image, gap: int = 2) -> Image.Image:
    """Patches physically separated by transparent gaps (W-MSA-window look).
    Canvas grows to SIZE + (GRID-1)*gap; gaps stay transparent for draw.io."""
    canvas = SIZE + (GRID - 1) * gap
    out = Image.new("RGBA", (canvas, canvas), (0, 0, 0, 0))
    src = img.convert("RGBA")
    for gy in range(GRID):
        for gx in range(GRID):
            tile = src.crop((gx * PATCH, gy * PATCH, (gx + 1) * PATCH, (gy + 1) * PATCH))
            out.paste(tile, (gx * (PATCH + gap), gy * (PATCH + gap)))
    return out


def attn_weights(attn, query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
    """Manual softmax(QK^T/sqrt(d)) replicating MultiHeadAttention projections
    (SDPA does not return weights). Returns (h, nq, nk)."""
    b, nq, d = query.shape
    nk = key.shape[1]
    h, hd = attn.num_heads, attn.head_dim
    q = attn.q_proj(query).view(b, nq, h, hd).transpose(1, 2)
    k = attn.k_proj(key).view(b, nk, h, hd).transpose(1, 2)
    w = (q @ k.transpose(-2, -1)) / math.sqrt(hd)
    return w.softmax(-1)[0]  # (h, nq, nk)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(CKPT, map_location=device)
    model = CCStyTr(**(state.get("model_cfg") or {})).to(device).eval()
    model.load_state_dict(state["model"] if "model" in state else state)
    print("loaded:", CKPT.name, "| smca_mode =", model.smca_mode)

    tf = test_transform(SIZE)
    content = tf(Image.open(CONTENT).convert("RGB")).unsqueeze(0).to(device)
    style = tf(Image.open(STYLE).convert("RGB")).unsqueeze(0).to(device)
    content_pil = to_pil(content[0].cpu())
    style_pil = to_pil(style[0].cpu())

    # ---- s1_01 / s1_02: raw inputs ----
    content_pil.save(OUT / "s1_01_content.png")
    style_pil.save(OUT / "s1_02_style.png")

    # ---- s1_03 / s1_04: patch grid (overlay) + separated variants ----
    draw_grid(content_pil).save(OUT / "s1_03_content_patches.png")
    draw_grid(style_pil).save(OUT / "s1_04_style_patches.png")
    draw_separated(content_pil).save(OUT / "s1_03b_content_separated.png")
    draw_separated(style_pil).save(OUT / "s1_04b_style_separated.png")

    # ---- s1_03c / s1_04c: per-token average color (patch embedding output as image) ----
    with torch.no_grad():
        ct_embed = model.content_embed(content)
        st_embed = model.style_embed(style)
        # Each token is 512-d; project to RGB via PCA and reshape back to spatial for visualization
        ct_proj = ct_embed[0].cpu().numpy()  # (N, 512)
        st_proj = st_embed[0].cpu().numpy()

        def tokens_to_image(tokens):
            grid = GRID
            # Reduce 512-d to single scalar per token via L2 norm, then stretch for visibility
            feat = np.linalg.norm(tokens, axis=1, keepdims=False) / np.sqrt(512)
            feat /= feat.max() + 1e-8
            img = feat.reshape(grid, grid)
            return Image.fromarray((255.0 * img).astype(np.uint8))

        to_im_ct = tokens_to_image(ct_proj)
        to_im_st = tokens_to_image(st_proj)
        to_im_ct.save(OUT / "s1_03c_content_tokens_avg.png")
        to_im_st.save(OUT / "s1_04c_style_tokens_avg.png")

        # ---- s1_03e / s1_04e: channel-wise variance projection (RGB-like)
        def tokens_to_rgb_via_variance(tokens):
            grid = GRID
            N = grid * grid
            feat = tokens[:N]  # (4096, 512)
            # Compute mean/variance over a subset of channels (first 64)
            subfeat = feat[:, :64]
            mean_per_token = subfeat.mean(axis=1, keepdims=True)  # (N, 1)
            var_per_token = subfeat.var(axis=1, keepdims=True)    # (N, 1)
            # Use mean, std, var as R,G,B channels
            r = (mean_per_token - mean_per_token.min()) / (mean_per_token.max() - mean_per_token.min() + 1e-8)
            g = (var_per_token - var_per_token.min()) / (var_per_token.max() - var_per_token.min() + 1e-8)
            b = np.zeros_like(r)
            b[:] = 0.5  # constant background offset
            img = np.stack([r, g, b], axis=-1).reshape(grid, grid, 3)
            # Upsample
            img_full = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)
            return Image.fromarray((img_full * 255).astype(np.uint8))

        try:
            rgb_ct = tokens_to_rgb_via_variance(ct_proj)
            rgb_st = tokens_to_rgb_via_variance(st_proj)
            rgb_ct.save(OUT / "s1_03e_content_pca_color.png")
            rgb_st.save(OUT / "s1_04e_style_pca_color.png")
        except Exception as e:
            print(f"RGB feature map skipped ({e})")
            rgb_ct = rgb_st = None

        # ---- s1_03d / s1_04d: patch-to-token schematic (illustrative, not data) ----
        def draw_patch_to_token(img_pil, name):
            # Draw a clean layout showing one patch -> 512-d vector projection
            import matplotlib.pyplot as plt
            fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(8, 2.2), dpi=200)

            src = np.asarray(img_pil.convert("RGB"))
            gx, gy = 5, 3
            patch = src[gy*PATCH:(gy+1)*PATCH, gx*PATCH:(gx+1)*PATCH].astype(np.float32) / 255.0

            ax0.imshow(patch)
            ax0.set_title("patch $x \\in \\mathbb{R}^{192}$", fontsize=10, ha="center")
            ax0.axis("off")

            ax1.text(0.5, 0.5, "W in R^(192x512)\n(learned weights)",
                     fontsize=14, va="center", ha="center")
            ax1.axis("off")

            ax2.text(0.5, 0.5, "h = ReLU(W.T x) in R^512\n(output token)",
                     fontsize=14, va="center", ha="center")
            ax2.axis("off")

            plt.savefig(OUT / name, bbox_inches="tight", pad_inches=0.1)
            plt.close()

        draw_patch_to_token(content_pil, "s1_03d_patch_embed_schematic.png")
        draw_patch_to_token(style_pil, "s1_04d_patch_embed_schematic.png")

    grid_hw = (GRID, GRID)
    with torch.no_grad():
        # ================= STAGE 1: ENCODING =================
        ct0 = model.content_embed(content)
        cape = model.cape(ct0, grid_hw)
        ct_in = ct0 + cape
        st0 = model.style_embed(style)

        # ---- s1_05: CAPE magnitude per token ----
        cape_mag = cape[0].norm(dim=-1).reshape(GRID, GRID).cpu().numpy()
        cape_mag /= cape_mag.max() + 1e-8
        overlay_heatmap(content_pil, cape_mag, alpha=0.45).save(OUT / "s1_05_cape.png")

        # encoder forward, capturing last-layer self-attention
        x = ct_in
        for i, layer in enumerate(model.content_encoder.layers):
            if i == len(model.content_encoder.layers) - 1:
                w = attn_weights(layer.self_attn, layer.norm1(x), layer.norm1(x))
            x = layer(x)
        ct = x

        ys = st0
        for i, layer in enumerate(model.style_encoder.layers):
            if i == len(model.style_encoder.layers) - 1:
                ws = attn_weights(layer.self_attn, layer.norm1(ys), layer.norm1(ys))
            ys = layer(ys)
        st = ys

        texture_tokens, color_mean, color_std = model.style_decouple(st)

        # saliency (token level, used by the gate)
        edge_map = model.edge(content)                     # (1,1,H,W)
        sal_tok = pool_map_to_tokens(edge_map, grid_hw)[0]  # (N,)
        sal_grid = sal_tok.reshape(GRID, GRID).cpu().numpy()

        # query tokens: most salient (text) vs least salient (background)
        q_text = int(sal_tok.argmax())
        q_bg = int(sal_tok.argmin())

        # ---- s1_06: content self-attention from text token ----
        amap = w[:, q_text, :].mean(0).reshape(GRID, GRID).cpu().numpy()
        amap /= amap.max() + 1e-8
        overlay_heatmap(content_pil, amap, alpha=0.55, cmap="magma").save(
            OUT / "s1_06_enc_selfattn_content.png")

        # ---- s1_07: style self-attention from most active style token ----
        q_style = int(st0[0].norm(dim=-1).argmax())
        smap = ws[:, q_style, :].mean(0).reshape(GRID, GRID).cpu().numpy()
        smap /= smap.max() + 1e-8
        overlay_heatmap(style_pil, smap, alpha=0.55, cmap="magma").save(
            OUT / "s1_07_enc_selfattn_style.png")

        # ================= STAGE 2: DECODING =================
        # ---- s2_01: Sobel saliency map ----
        sob = edge_map[0, 0].cpu().numpy()
        save_heatmap_only(sob, "s2_01_sobel.png", cmap="gray")

        # ---- s2_02: token-level saliency overlay ----
        overlay_heatmap(content_pil, sal_grid, alpha=0.5).save(
            OUT / "s2_02_saliency_tokens.png")

        # decoder forward, capturing per-layer gate maps and cross-attention
        x = ct
        gates, crossw = [], []
        for layer in model.fusion_decoder.layers:
            x = x + layer.dropout(layer.self_attn(layer.norm1(x), layer.norm1(x), layer.norm1(x)))
            g = (1.0 - layer.gamma.abs() * sal_tok).clamp(0.0, 1.0)
            gates.append(g.reshape(GRID, GRID).cpu().numpy())
            crossw.append(attn_weights(layer.cross_attn, layer.norm2(x), texture_tokens))
            attn_out = layer.cross_attn(layer.norm2(x), texture_tokens, texture_tokens)
            x = x + layer.dropout(g.unsqueeze(-1).to(attn_out.dtype).unsqueeze(0) * attn_out)
            x = x + layer.dropout(layer.ffn(layer.norm3(x)))
        fused = x

        # ---- s2_03/04/05: per-layer gate maps ----
        for i, g in enumerate(gates, 1):
            overlay_heatmap(content_pil, g, alpha=0.5, cmap="RdYlGn").save(
                OUT / f"s2_0{2+i}_gate_l{i}.png")

        # ---- s2_06/07: cross-attention (last layer) over style tokens ----
        wl = crossw[-1]  # (h, Nc, Ns)
        for name, qi in [("s2_06_crossattn_text.png", q_text),
                         ("s2_07_crossattn_bg.png", q_bg)]:
            m = wl[:, qi, :].mean(0).reshape(GRID, GRID).cpu().numpy()
            m /= m.max() + 1e-8
            overlay_heatmap(style_pil, m, alpha=0.55, cmap="magma").save(OUT / name)

        # ---- s2_08: style color statistics (mu / sigma bars) ----
        mu = color_mean[0, 0].cpu().numpy()
        sd = color_std[0, 0].cpu().numpy()
        bar = np.stack([mu, sd])  # (2, 512)
        bar = (bar - bar.min()) / (bar.max() - bar.min() + 1e-8)
        bar_img = np.repeat(np.repeat(bar, 32, axis=0), 1, axis=1)  # (64,512)
        colored = (matplotlib.colormaps["viridis"](bar_img)[..., :3] * 255).astype(np.uint8)
        Image.fromarray(colored).resize((SIZE, 128), Image.NEAREST).save(
            OUT / "s2_08_color_stats.png")

        # ---- s2_09: PCA projection of fused tokens ----
        f = fused[0]  # (N, dim)
        _, _, v = torch.pca_lowrank(f - f.mean(0, keepdim=True), q=3)
        proj = (f - f.mean(0, keepdim=True)) @ v  # (N,3)
        proj = proj.reshape(GRID, GRID, 3).cpu().numpy()
        proj = (proj - proj.min(axis=(0, 1))) / (proj.max(axis=(0, 1)) - proj.min(axis=(0, 1)) + 1e-8)
        Image.fromarray((proj * 255).astype(np.uint8)).resize(
            (SIZE, SIZE), Image.BILINEAR).save(OUT / "s2_09_fused_pca.png")

        # ---- s2_10: demo alpha map (alpha = 1 - 0.7 * S) ----
        sob_t = torch.from_numpy(sob).to(device)
        alpha_demo = (1.0 - 0.7 * sob_t).clamp(0.0, 1.0)
        save_heatmap_only(alpha_demo.cpu().numpy(), "s2_10_alpha_demo.png", cmap="RdBu_r")

        # ---- s2_11/12/13: uniform alpha outputs; s2_14: demo alpha output ----
        for name, aval in [("s2_11_out_a000.png", 0.0),
                           ("s2_12_out_a050.png", 0.5),
                           ("s2_13_out_a100.png", 1.0)]:
            amap_t = torch.full((1, 1, SIZE, SIZE), aval, device=device)
            out = model(content, style, amap_t)
            to_pil(out[0].cpu()).save(OUT / name)
        out_demo = model(content, style, alpha_demo.unsqueeze(0).unsqueeze(0))
        to_pil(out_demo[0].cpu()).save(OUT / "s2_14_out_demo.png")

    print("done ->", OUT)
    for p in sorted(OUT.glob("*.png")):
        print(" ", p.name)

    # ---- EXTRA: Generate CAPE map if style image is treated as content ----
    try:
        bird_pil = Image.open(ROOT / "data/eval/featured/style_flat/western_painting__met_406721.png")
        bird_pil = bird_pil.resize((SIZE, SIZE), Image.Resampling.LANCZOS)
        bird_np = np.array(bird_pil)  # (H,W,3)
        bird_tensor = torch.from_numpy(bird_np).permute(2, 0, 1).float() / 255.0  # (3,H,W)
        bird_img = bird_tensor.unsqueeze(0).to(device)  # (1,3,H,W)
        
        with torch.no_grad():
            bird_ct0 = model.content_embed(bird_img)
            bird_cape = model.cape(bird_ct0, grid_hw)
            cape_mag = bird_cape[0].norm(dim=-1).reshape(GRID, GRID).cpu().numpy()
            cape_mag /= cape_mag.max() + 1e-8
            # Direct grayscale heatmap save
            import matplotlib.pyplot as plt
            cmap = plt.get_cmap("Greens")
            cape_vis = (cmap(cape_mag)[..., :3] * 255).astype(np.uint8)
            Image.fromarray(cape_vis).save(OUT / "s1_05b_style_as_content_cape.png")
            print("  s1_05b_style_as_content_cape.png (extra demo)")
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"Extra CAPE demo skipped ({e})")


if __name__ == "__main__":
    main()
