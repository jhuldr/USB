import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils.denoise import one_step_denoise


def get_middle_slice(vol, mid_slice=60):
    vol = vol[0, 0]
    slice_2d = np.rot90(vol[:, mid_slice, :])
    slice_norm = (slice_2d - slice_2d.min()) / (slice_2d.max() - slice_2d.min() + 1e-8)
    return (slice_norm * 255).astype(np.uint8)

def visualize_diffusion_batch(
    x0, y0, a, e_x, e_y, t, betas,
    vae, batch,
    epoch, log_dir, train_config, device,
    model, x0_pred=None
):
    """Visualize the diffusion process for a batch."""

    vae = vae.to(device)
    x0, y0 = x0.to(device), y0.to(device)
    a = a.to(device)
    e_x, e_y = e_x.to(device), e_y.to(device)
    t = t.to(device)

    xt = x0 * a.sqrt() + e_x * (1.0 - a).sqrt()
    yt = y0 * a.sqrt() + e_y * (1.0 - a).sqrt()

    if train_config.mode == 'lesion':
        x0_pred_model = one_step_denoise(model, xt, yt, t, betas)

    elif train_config.mode == 'brain':
        assert x0_pred is not None, "x0_pred must be provided for brain model."
        y0_pred_model = one_step_denoise(model, yt, x0_pred, t, betas)
    
    def decode_latent(z):
        z = z / 0.18215
        img = vae.decode([z])[0]._sample()
        return img.detach().cpu().numpy()

    for i in tqdm(range(x0.shape[0]), file=sys.__stdout__):

        # Decode individual samples
        x0_img = decode_latent(x0[i:i+1])
        y0_img = decode_latent(y0[i:i+1])
        xt_img = decode_latent(xt[i:i+1])
        yt_img = decode_latent(yt[i:i+1])

        if train_config.mode == 'lesion':
            x0_pred_img = decode_latent(x0_pred_model[i:i+1])

        elif train_config.mode == 'brain':
            x0_pred_img = decode_latent(x0_pred[i:i+1])
            y0_pred_img = decode_latent(y0_pred_model[i:i+1])

        # Diffusion slices
        slice_x0 = get_middle_slice(x0_img)
        slice_y0 = get_middle_slice(y0_img)
        slice_xt = get_middle_slice(xt_img)
        slice_yt = get_middle_slice(yt_img)

        if train_config.mode == 'lesion':
            slice_pred_x0 = get_middle_slice(x0_pred_img)

        elif train_config.mode == 'brain':
            slice_pred_x0 = get_middle_slice(x0_pred_img)
            slice_pred_y0 = get_middle_slice(y0_pred_img)

        vis_dir = os.path.join(log_dir, "visualization")
        os.makedirs(vis_dir, exist_ok=True)

        img_name = batch['img_name'][i]
        pathol_name = batch['pathol_name'][i]
        time_t = t[i].item()

        png_path = os.path.join(
            vis_dir,
            f"epoch{epoch}_sample_{i}_img_{img_name}_pathol_{pathol_name}_time_{time_t}.png"
        )

        # === Pretty Visualization ===
        if train_config.mode == 'lesion':
            images = [
                slice_x0, slice_y0,
                slice_xt, slice_yt,
                slice_pred_x0
            ]

            titles = [
                "x0", "y0",
                "xt", "yt",
                "x0_pred"
            ]

        else:  # brain mode
            images = [
                slice_x0, slice_y0,
                slice_xt, slice_yt,
                slice_pred_x0, slice_pred_y0
            ]

            titles = [
                "x0", "y0",
                "xt", "yt",
                "x0_pred", "y0_pred"
            ]

        # Number of columns = number of images
        num_cols = len(images)
        fig, axes = plt.subplots(1, num_cols, figsize=(3*num_cols, 3))

        if num_cols == 1:
            axes = [axes]  # ensure iterable

        for ax, img, title in zip(axes, images, titles):
            ax.imshow(img, cmap="gray")
            ax.set_title(title, fontsize=10)
            ax.axis("off")

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(png_path, dpi=150)
        plt.close(fig)
