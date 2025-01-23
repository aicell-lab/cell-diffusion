import os
import glob
import torch
import wandb

from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from torchvision.utils import make_grid
from torchvision.transforms.functional import to_pil_image

from diffusers import AutoencoderKL, UNet2DModel, DDPMScheduler
import torch.optim as optim
import torch.nn.functional as F


class ImageNetSubsetDataset(Dataset):
    def __init__(self, root, image_size=512):
        self.files = sorted(glob.glob(os.path.join(root, "*", "*.JPEG")))
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = self.files[idx]
        img = Image.open(img_path).convert("RGB")
        return self.transform(img)


def sample_images(unet, vae, scheduler, device, num_samples=4, num_inference_steps=50):
    unet.eval()
    with torch.no_grad():
        latents = torch.randn((num_samples, 4, 64, 64), device=device)
        scheduler.set_timesteps(num_inference_steps)
        for t in scheduler.timesteps:
            model_out = unet(latents, t).sample
            latents = scheduler.step(model_out, t, latents).prev_sample
        images = vae.decode(latents / 0.18215).sample

    images = (images * 0.5 + 0.5).clamp(0, 1)
    unet.train()
    return images


def compute_val_loss(unet, vae, scheduler, val_loader, device):
    unet.eval()
    total_loss = 0.0
    total_steps = 0
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            latents = vae.encode(batch).latent_dist.sample() * 0.18215
            bsz = latents.shape[0]
            timesteps = torch.randint(
                0, scheduler.num_train_timesteps, (bsz,), device=device
            ).long()
            noise = torch.randn_like(latents)
            noisy_latents = scheduler.add_noise(latents, noise, timesteps)

            pred = unet(noisy_latents, timesteps).sample
            loss = F.mse_loss(pred, noise)
            total_loss += loss.item()
            total_steps += 1

    unet.train()
    return total_loss / max(1, total_steps)


def train_unconditional_imagenet(
    data_root="datasets/imagenet/train.X1",
    val_root=None,
    output_dir="imagenet_uncond_checkpoints",
    epochs=10,
    batch_size=16,
    lr=1e-4,
    resume_checkpoint=None,
):
    """
    Trains an unconditional diffusion model on an ImageNet subset.
    Arguments:
      data_root: Path to training images.
      val_root:  Path to validation images (optional).
      output_dir: Where to save outputs.
      epochs: Number of training epochs.
      batch_size: Batch size.
      lr: Learning rate.
      resume_checkpoint: path to a .pt checkpoint file to resume from (optional).
    """
    wandb.init(
        project="diffusion", entity="aicellio", name="imagenet-unconditional-run"
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    os.makedirs(output_dir, exist_ok=True)

    # 1) Load VAE
    vae = AutoencoderKL.from_pretrained(
        "CompVis/stable-diffusion-v1-4", subfolder="vae"
    ).to(device)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)

    # 2) Create U-Net
    unet = UNet2DModel(
        sample_size=64,
        in_channels=4,
        out_channels=4,
        layers_per_block=2,
        block_out_channels=(128, 256),
        down_block_types=("DownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "UpBlock2D"),
    ).to(device)

    scheduler = DDPMScheduler(num_train_timesteps=1000)

    # 3) Load datasets
    train_dataset = ImageNetSubsetDataset(root=data_root, image_size=512)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )

    # Optional validation set
    val_loader = None
    if val_root is not None:
        val_dataset = ImageNetSubsetDataset(root=val_root, image_size=512)
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, drop_last=True
        )

    # 4) Optimizer
    optimizer = optim.AdamW(unet.parameters(), lr=lr)

    global_step = 0
    start_epoch = 0

    # 5) Resume from checkpoint if provided
    if resume_checkpoint is not None and os.path.isfile(resume_checkpoint):
        print(f"Resuming from checkpoint: {resume_checkpoint}")
        checkpoint_data = torch.load(resume_checkpoint, map_location="cpu")
        unet.load_state_dict(checkpoint_data["unet"])
        optimizer.load_state_dict(checkpoint_data["optimizer"])
        start_epoch = checkpoint_data["epoch"] + 1
        global_step = checkpoint_data["global_step"]

    # 6) Training loop
    for epoch in range(start_epoch, epochs):
        for step, batch in enumerate(train_loader):
            batch = batch.to(device)

            with torch.no_grad():
                latents = vae.encode(batch).latent_dist.sample() * 0.18215

            bsz = latents.shape[0]
            timesteps = torch.randint(
                0, scheduler.num_train_timesteps, (bsz,), device=device
            ).long()
            noise = torch.randn_like(latents)
            noisy_latents = scheduler.add_noise(latents, noise, timesteps)

            pred = unet(noisy_latents, timesteps).sample
            loss = F.mse_loss(pred, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            wandb.log({"train/loss": loss.item()}, step=global_step)
            global_step += 1

            # Print every 50 steps
            if step % 50 == 0:
                print(
                    f"Epoch {epoch} | Step {step} | Loss {loss.item():.4f} | Global Step {global_step}"
                )

            # Sample images every 1000 steps
            if global_step % 1000 == 0:
                images = sample_images(
                    unet, vae, scheduler, device, num_samples=4, num_inference_steps=50
                )
                grid = make_grid(images, nrow=2)
                grid_pil = to_pil_image(grid)
                wandb.log(
                    {
                        "sample_images": wandb.Image(
                            grid_pil, caption=f"Step {global_step}"
                        )
                    },
                    step=global_step,
                )

        # End of epoch: validation & checkpoint
        if val_loader is not None:
            val_loss = compute_val_loss(unet, vae, scheduler, val_loader, device)
            wandb.log({"val/loss": val_loss, "epoch": epoch}, step=global_step)
            print(f"Validation Loss after epoch {epoch}: {val_loss:.4f}")

        # Save a checkpoint each epoch
        checkpoint_dict = {
            "unet": unet.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
        }
        save_path = os.path.join(output_dir, f"unet_epoch_{epoch}.pt")
        torch.save(checkpoint_dict, save_path)
        print(f"Saved checkpoint to {save_path}")


if __name__ == "__main__":
    train_unconditional_imagenet()
