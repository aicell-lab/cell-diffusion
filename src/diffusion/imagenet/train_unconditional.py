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
    """
    Scans the 'train.X1' (or similar) folder which contains subfolders for each class.
    E.g.,
      datasets/imagenet/train.X1/
        n01440764/  *.JPEG
        n01484850/  *.JPEG
        ...
    We'll resize each image to 512x512 for the SD VAE.
    """

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

    # Convert [-1,1] to [0,1] for display
    images = (images * 0.5 + 0.5).clamp(0, 1)
    unet.train()
    return images


def train_unconditional_imagenet(
    data_root="datasets/imagenet/train.X1",
    output_dir="imagenet_uncond_checkpoints",
    epochs=10,
    batch_size=4,
    lr=1e-4,
):
    wandb.init(
        project="diffusion", entity="aicellio", name="imagenet-unconditional-run"
    )

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    os.makedirs(output_dir, exist_ok=True)

    vae = AutoencoderKL.from_pretrained(
        "CompVis/stable-diffusion-v1-4", subfolder="vae"
    ).to(device)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)

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

    dataset = ImageNetSubsetDataset(root=data_root, image_size=512)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )

    optimizer = optim.AdamW(unet.parameters(), lr=lr)

    global_step = 0

    for epoch in range(epochs):
        for step, batch in enumerate(dataloader):
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

            wandb.log({"train/loss": loss.item(), "step": global_step})
            global_step += 1

            if step % 50 == 0:
                print(f"Epoch {epoch} | Step {step} | Loss {loss.item():.4f}")

        images = sample_images(
            unet, vae, scheduler, device, num_samples=4, num_inference_steps=50
        )
        grid = make_grid(images, nrow=2)
        grid_pil = to_pil_image(grid)

        grid_path = os.path.join(output_dir, f"samples_epoch_{epoch}.png")
        grid_pil.save(grid_path)

        wandb.log(
            {
                "epoch": epoch,
                "sample_images": wandb.Image(grid_pil, caption=f"Epoch {epoch}"),
            }
        )

        if epoch % 2 == 0:
            torch.save(
                unet.state_dict(), os.path.join(output_dir, f"unet_epoch_{epoch}.pt")
            )


if __name__ == "__main__":
    train_unconditional_imagenet()
