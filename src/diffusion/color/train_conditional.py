import os
import glob
import random
import torch
import torch.optim as optim
import torch.nn.functional as F
import wandb


from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from torchvision.utils import make_grid
from torchvision.transforms.functional import to_pil_image

from diffusers import (
    UNet2DConditionModel,
    DDPMScheduler,
    AutoencoderKL,
)


class ColorOneHotDataset(Dataset):
    def __init__(self, root, color_to_id, image_size=256):
        self.root = root
        self.color_to_id = color_to_id
        self.num_colors = len(color_to_id)

        self.image_paths = []
        self.labels = []

        for color_folder in sorted(os.listdir(root)):
            subdir = os.path.join(root, color_folder)
            if not os.path.isdir(subdir):
                continue
            color_id = color_to_id[color_folder]
            image_files = sorted(glob.glob(os.path.join(subdir, "*")))

            for img_path in image_files:
                self.image_paths.append(img_path)
                self.labels.append(color_id)

        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        color_id = self.labels[idx]
        one_hot_vec = torch.zeros(self.num_colors)
        one_hot_vec[color_id] = 1.0

        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        return img, one_hot_vec


def create_small_unet(num_colors):
    """
    A smaller U-Net with cross-attention in all blocks, for CFG + one-hot.
    """
    model = UNet2DConditionModel(
        sample_size=64,
        in_channels=4,
        out_channels=4,
        down_block_types=(
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
        ),
        mid_block_type="UNetMidBlock2DCrossAttn",
        up_block_types=(
            "CrossAttnUpBlock2D",
            "CrossAttnUpBlock2D",
        ),
        block_out_channels=(128, 256),
        layers_per_block=2,
        cross_attention_dim=num_colors,  # same as number of classes for one-hot
    )
    return model


def validate_one_epoch(unet, vae, scheduler, val_loader, device, cfg_prob=0.0):
    """
    We do a single pass noise-prediction with random label dropout (cfg_prob).
    """
    unet.eval()
    total_loss = 0.0
    total_steps = 0

    with torch.no_grad():
        for images, one_hot_vec in val_loader:
            images = images.to(device)
            one_hot_vec = one_hot_vec.to(device)

            latents = vae.encode(images).latent_dist.sample() * 0.18215
            batch_size = latents.shape[0]

            # Possibly drop label => unconditional
            for i in range(batch_size):
                if random.random() < cfg_prob:
                    one_hot_vec[i] = 0.0

            # [batch_size, 1, num_colors]
            cond_emb = one_hot_vec.unsqueeze(1)

            timesteps = torch.randint(
                0, scheduler.num_train_timesteps, (batch_size,), device=device
            ).long()
            noise = torch.randn_like(latents)
            noisy_latents = scheduler.add_noise(latents, noise, timesteps)

            model_out = unet(
                noisy_latents, timesteps, encoder_hidden_states=cond_emb
            ).sample
            loss = F.mse_loss(model_out, noise)

            total_loss += loss.item()
            total_steps += 1

    unet.train()
    return total_loss / max(1, total_steps)


def train_conditional_one_hot_cfg(
    data_root="datasets/colors_rgb",
    output_dir="color_diffusion_cond_checkpoints_one_hot_cfg",
    epochs=25,
    batch_size=4,
    lr=1e-4,
    cfg_prob=0.1,  # fraction of label-drop in training
    val_cfg_prob=0.0,  # label-drop in validation
    guidance_scale=7.5,  # used for sampling
):
    wandb.init(
        project="diffusion", entity="aicellio", name="color-diffusion-one-hot-cfg"
    )

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    os.makedirs(output_dir, exist_ok=True)

    train_dir = os.path.join(data_root, "train")
    val_dir = os.path.join(data_root, "val")

    # Build color->id mapping from train folders
    color_folders = [
        d
        for d in sorted(os.listdir(train_dir))
        if os.path.isdir(os.path.join(train_dir, d))
    ]
    color_to_id = {c: i for i, c in enumerate(color_folders)}
    id_to_color = {i: c for c, i in color_to_id.items()}
    num_colors = len(color_to_id)
    print(f"color_to_id: {color_to_id}")

    vae = AutoencoderKL.from_pretrained(
        "CompVis/stable-diffusion-v1-4", subfolder="vae"
    ).to(device)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)

    unet = create_small_unet(num_colors).to(device)
    scheduler = DDPMScheduler(num_train_timesteps=1000)

    train_dataset = ColorOneHotDataset(train_dir, color_to_id)
    val_dataset = ColorOneHotDataset(val_dir, color_to_id)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, drop_last=True
    )

    optimizer = optim.AdamW(unet.parameters(), lr=lr)

    global_step = 0

    for epoch in range(epochs):
        unet.train()
        for step, (images, one_hot_vec) in enumerate(train_loader):
            images = images.to(device)
            one_hot_vec = one_hot_vec.to(device)

            with torch.no_grad():
                latents = vae.encode(images).latent_dist.sample() * 0.18215

            # Classifier-Free Guidance (training):
            # randomly drop label => unconditional
            for i in range(latents.shape[0]):
                if random.random() < cfg_prob:
                    one_hot_vec[i] = 0.0

            # shape [batch_size, 1, num_colors]
            cond_emb = one_hot_vec.unsqueeze(1)

            bsz = latents.shape[0]
            timesteps = torch.randint(
                0, scheduler.num_train_timesteps, (bsz,), device=device
            ).long()
            noise = torch.randn_like(latents)
            noisy_latents = scheduler.add_noise(latents, noise, timesteps)

            model_out = unet(noisy_latents, timesteps, encoder_hidden_states=cond_emb)
            pred = model_out.sample

            loss = F.mse_loss(pred, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            wandb.log({"train/loss": loss.item(), "step": global_step})
            global_step += 1

            if step % 50 == 0:
                print(f"Epoch {epoch} | Step {step} | Loss {loss.item():.4f}")

        # Validation
        val_loss = validate_one_epoch(
            unet, vae, scheduler, val_loader, device, cfg_prob=val_cfg_prob
        )
        wandb.log({"val/loss": val_loss, "epoch": epoch})
        print(f"Epoch {epoch} | val_loss: {val_loss:.4f}")

        # Sample images with CFG
        random_label_id = random.randint(0, num_colors - 1)
        random_color = id_to_color[random_label_id]
        images = sample_one_hot_cfg(
            unet,
            vae,
            scheduler,
            label_id=random_label_id,
            guidance_scale=guidance_scale,
            num_samples=4,
            num_inference_steps=30,
            device=device,
        )
        grid = make_grid(images, nrow=2)
        grid_pil = to_pil_image(grid)

        grid_path = os.path.join(output_dir, f"samples_epoch_{epoch}.png")
        grid_pil.save(grid_path)

        wandb.log(
            {
                "epoch": epoch,
                "sample_images": wandb.Image(
                    grid_pil, caption=f"Epoch {epoch} - {random_color}"
                ),
            }
        )

        # Save checkpoint
        if epoch % 5 == 0:
            ckpt_path = os.path.join(output_dir, f"unet_epoch_{epoch}.pt")
            torch.save(unet.state_dict(), ckpt_path)
            print(f"Saved checkpoint to {ckpt_path}")


# ---------------------------
# CFG Sampling
# ---------------------------
def sample_one_hot_cfg(
    unet,
    vae,
    scheduler,
    label_id,
    guidance_scale=7.5,
    num_samples=4,
    num_inference_steps=50,
    device="cuda",
):
    """
    Classifier-Free Guidance at inference:
      1) uncond_out = unet(noisy_latents, t, zero_vector)
      2) cond_out   = unet(noisy_latents, t, one_hot_vector)
      3) final_out  = uncond_out + scale * (cond_out - uncond_out)
    """
    unet.eval()
    with torch.no_grad():
        # Prepare unconditional (empty) and conditional (one-hot) embeddings
        num_colors = unet.config.cross_attention_dim
        # cond: shape [num_samples, num_colors]
        cond = torch.zeros(num_samples, num_colors, device=device)
        cond[range(num_samples), label_id] = 1.0
        cond = cond.unsqueeze(1)  # [bs, 1, num_colors]

        # uncond: all zeros
        uncond = torch.zeros_like(cond)

        latents = torch.randn((num_samples, 4, 64, 64), device=device)
        scheduler.set_timesteps(num_inference_steps)

        for t in scheduler.timesteps:
            # 1) Unconditional pass
            uncond_out = unet(latents, t, encoder_hidden_states=uncond).sample
            # 2) Conditional pass
            cond_out = unet(latents, t, encoder_hidden_states=cond).sample

            # 3) Combine
            cfg_out = uncond_out + guidance_scale * (cond_out - uncond_out)

            latents = scheduler.step(cfg_out, t, latents).prev_sample

        images = vae.decode(latents / 0.18215).sample

    images = (images * 0.5 + 0.5).clamp(0, 1)
    return images


# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    train_conditional_one_hot_cfg()
