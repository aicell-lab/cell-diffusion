import os
import glob
import random
import torch
import wandb

from torch import nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from torchvision.utils import make_grid
from torchvision.transforms.functional import to_pil_image

from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL

import torch.optim as optim
import torch.nn.functional as F


class ColorLabelDataset(Dataset):
    """
    Directory structure example:
        datasets/colors/train/
            red/*.png
            blue/*.png
            ...
        datasets/colors/val/
            red/*.png
            blue/*.png
            ...
    We map each folder name (e.g. "red") to an integer label ID,
    and store that in self.label_to_id and self.id_to_label.
    """

    def __init__(self, root, color_to_id, image_size=256):
        self.root = root
        self.color_to_id = color_to_id  # dict like {"black":0, "blue":1, ...}
        self.image_paths = []
        self.labels = []

        # Each subfolder is a color name
        for color_folder in sorted(os.listdir(root)):
            subdir = os.path.join(root, color_folder)
            if not os.path.isdir(subdir):
                continue
            label_id = color_to_id[color_folder]
            image_files = sorted(glob.glob(os.path.join(subdir, "*")))
            for img_path in image_files:
                self.image_paths.append(img_path)
                self.labels.append(label_id)

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
        label_id = self.labels[idx]

        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        return img, label_id


class LabelEmbedding(nn.Module):
    """
    A simple embedding layer for color labels.
    We produce shape [batch_size, 1, embed_dim] for cross-attention.
    """

    def __init__(self, num_labels, embed_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(num_labels, embed_dim)

    def forward(self, label_ids):
        # label_ids: shape [batch_size]
        # output: [batch_size, 1, embed_dim]
        embeds = self.embedding(label_ids)  # [batch_size, embed_dim]
        embeds = embeds.unsqueeze(1)  # add seq_len=1 dimension
        return embeds


def sample_images(
    unet,
    vae,
    scheduler,
    label_embedder,
    device,
    label_id=0,
    embed_dim=64,
    num_samples=4,
    num_inference_steps=50,
):
    """
    Generate images using a particular color label_id.
    """
    unet.eval()
    label_embedder.eval()

    with torch.no_grad():
        # Create repeated label IDs for all samples
        label_ids = torch.tensor([label_id] * num_samples, device=device)

        # Convert label IDs to embedding
        cond_emb = label_embedder(label_ids)  # shape [num_samples, 1, embed_dim]

        # Start from random noise in latent space
        latents = torch.randn((num_samples, 4, 64, 64), device=device)
        scheduler.set_timesteps(num_inference_steps)

        for t in scheduler.timesteps:
            model_out = unet(latents, t, encoder_hidden_states=cond_emb).sample
            latents = scheduler.step(model_out, t, latents).prev_sample

        images = vae.decode(latents / 0.18215).sample

    images = (images * 0.5 + 0.5).clamp(0, 1)
    unet.train()
    label_embedder.train()
    return images


def validate_one_epoch(unet, vae, scheduler, label_embedder, val_loader, device):
    """
    Basic validation loop computing average MSE noise prediction loss.
    """
    unet.eval()
    label_embedder.eval()
    total_loss = 0.0
    total_steps = 0

    with torch.no_grad():
        for images, label_ids in val_loader:
            images = images.to(device)
            label_ids = label_ids.to(device)
            latents = vae.encode(images).latent_dist.sample() * 0.18215

            cond_emb = label_embedder(label_ids)  # [bs, 1, embed_dim]

            bsz = latents.shape[0]
            timesteps = torch.randint(
                0, scheduler.num_train_timesteps, (bsz,), device=device
            ).long()
            noise = torch.randn_like(latents)
            noisy_latents = scheduler.add_noise(latents, noise, timesteps)

            model_out = unet(noisy_latents, timesteps, encoder_hidden_states=cond_emb)
            pred = model_out.sample

            loss = F.mse_loss(pred, noise)
            total_loss += loss.item()
            total_steps += 1

    unet.train()
    label_embedder.train()
    return total_loss / max(1, total_steps)


def train_conditional_simple(
    data_root="datasets/colors",  # expects train/ and val/
    output_dir="color_diffusion_cond_checkpoints",
    epochs=20,
    batch_size=4,
    lr=1e-4,
    embed_dim=64,
):
    """
    A simpler label-based approach:
    - We discover color folders in 'train' to build color->id mapping.
    - We embed that ID into a small learned vector for cross-attention.
    - We do a standard diffusion training loop with MSE loss on noise prediction.
    """

    wandb.init(project="diffusion", entity="aicellio", name="color-diffusion-simple")

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

    # 1) Build color->id mapping from train subfolders
    color_folders = [
        d
        for d in sorted(os.listdir(train_dir))
        if os.path.isdir(os.path.join(train_dir, d))
    ]
    color_to_id = {c: i for i, c in enumerate(color_folders)}
    id_to_color = {i: c for c, i in color_to_id.items()}
    num_colors = len(color_folders)
    print(f"Color -> ID mapping: {color_to_id}")

    # 2) Load VAE
    vae = AutoencoderKL.from_pretrained(
        "CompVis/stable-diffusion-v1-4", subfolder="vae"
    ).to(device)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)

    # 3) Create UNet2DConditionModel with cross_attention_dim=embed_dim
    unet = UNet2DConditionModel(
        sample_size=64,
        in_channels=4,
        out_channels=4,
        cross_attention_dim=embed_dim,
        layers_per_block=2,
        # e.g. block_out_channels=(320, 640, 640, 1280),
        # or leave defaults if you want a smaller model
    ).to(device)

    # 4) Create a small embedding layer for color labels
    label_embedder = LabelEmbedding(num_labels=num_colors, embed_dim=embed_dim).to(
        device
    )

    # 5) Scheduler
    scheduler = DDPMScheduler(num_train_timesteps=1000)

    # 6) Datasets & Dataloaders
    train_dataset = ColorLabelDataset(train_dir, color_to_id)
    val_dataset = ColorLabelDataset(val_dir, color_to_id)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, drop_last=True
    )

    # 7) Optimizer (includes UNet + label_embedder)
    optimizer = optim.AdamW(
        list(unet.parameters()) + list(label_embedder.parameters()), lr=lr
    )

    global_step = 0

    for epoch in range(epochs):
        unet.train()
        label_embedder.train()

        for step, (images, label_ids) in enumerate(train_loader):
            images = images.to(device)
            label_ids = label_ids.to(device)

            with torch.no_grad():
                latents = vae.encode(images).latent_dist.sample() * 0.18215

            # Convert label IDs to embeddings
            cond_emb = label_embedder(label_ids)  # shape [bs, 1, embed_dim]

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
            unet, vae, scheduler, label_embedder, val_loader, device
        )
        wandb.log({"val/loss": val_loss, "epoch": epoch})
        print(f"Epoch {epoch} | Validation Loss: {val_loss:.4f}")

        # Sample images from a random color
        random_label_id = random.randint(0, num_colors - 1)
        sampled_imgs = sample_images(
            unet,
            vae,
            scheduler,
            label_embedder,
            device,
            label_id=random_label_id,
            embed_dim=embed_dim,
            num_samples=4,
            num_inference_steps=30,
        )
        grid = make_grid(sampled_imgs, nrow=2)
        grid_pil = to_pil_image(grid)
        color_name = id_to_color[random_label_id]

        grid_path = os.path.join(output_dir, f"samples_epoch_{epoch}.png")
        grid_pil.save(grid_path)

        wandb.log(
            {
                "epoch": epoch,
                "sample_images": wandb.Image(
                    grid_pil, caption=f"Epoch {epoch} - Color: {color_name}"
                ),
            }
        )

        # Save checkpoint
        ckpt_path = os.path.join(output_dir, f"unet_epoch_{epoch}.pt")
        torch.save(
            {
                "unet": unet.state_dict(),
                "label_embedder": label_embedder.state_dict(),
            },
            ckpt_path,
        )
        print(f"Saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    train_conditional_simple()
