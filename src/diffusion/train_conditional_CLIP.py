import os
import glob
import random
import torch
import wandb

from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from torchvision.utils import make_grid
from torchvision.transforms.functional import to_pil_image

from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer
import torch.optim as optim
import torch.nn.functional as F


class ColorDatasetWithText(Dataset):
    """
    root/
      red/*.png
      green/*.png
      ...
    We'll create a textual "caption" based on the folder name.
    """

    def __init__(self, root, image_size=256):
        self.image_paths = []
        self.labels = []

        folders = sorted(glob.glob(os.path.join(root, "*")))
        for folder in folders:
            if not os.path.isdir(folder):
                continue
            label = os.path.basename(folder)
            image_files = sorted(glob.glob(os.path.join(folder, "*")))
            for img in image_files:
                self.image_paths.append(img)
                self.labels.append(label)

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
        label = self.labels[idx]
        caption = f"a photo of a {label} color"
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        return img, caption


def sample_images_cfg(
    unet,
    vae,
    scheduler,
    text_encoder,
    tokenizer,
    device,
    prompt="a photo of a red color",
    guidance_scale=7.5,
    num_samples=4,
    num_inference_steps=50,
):
    """
    Use classifier-free guidance for sampling.
    We'll do two passes each step:
      1) conditional:   unet(latents, t, text_emb)
      2) unconditional: unet(latents, t, empty_emb)
    Combine: uncond + guidance_scale * (cond - uncond)
    """

    unet.eval()
    text_encoder.eval()

    with torch.no_grad():
        # Encode the real prompt
        text_inputs_cond = tokenizer(
            [prompt] * num_samples,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).to(device)
        cond_emb = text_encoder(text_inputs_cond.input_ids).last_hidden_state

        # Encode the empty prompt
        text_inputs_uncond = tokenizer(
            ["" for _ in range(num_samples)],
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).to(device)
        uncond_emb = text_encoder(text_inputs_uncond.input_ids).last_hidden_state

        latents = torch.randn((num_samples, 4, 64, 64), device=device)
        scheduler.set_timesteps(num_inference_steps)

        for t in scheduler.timesteps:
            # 1) Unconditional pass
            uncond_out = unet(latents, t, encoder_hidden_states=uncond_emb).sample
            # 2) Conditional pass
            cond_out = unet(latents, t, encoder_hidden_states=cond_emb).sample

            cfg_out = uncond_out + guidance_scale * (cond_out - uncond_out)

            latents = scheduler.step(cfg_out, t, latents).prev_sample

        images = vae.decode(latents / 0.18215).sample

    images = (images * 0.5 + 0.5).clamp(0, 1)
    unet.train()
    text_encoder.train()
    return images, prompt


def validate_one_epoch(
    unet,
    vae,
    scheduler,
    text_encoder,
    tokenizer,
    val_loader,
    device,
    guidance_prob=0.1,
):
    """
    Compute average validation loss.
    We also apply random prompt dropout in validation, but you can set guidance_prob=0 if you prefer pure condition.
    """
    unet.eval()
    text_encoder.eval()
    total_loss = 0.0
    total_steps = 0

    with torch.no_grad():
        for images, captions in val_loader:
            images = images.to(device)
            latents = vae.encode(images).latent_dist.sample() * 0.18215

            # Possibly drop text for some fraction of examples (same as training)
            dropped_captions = []
            for c in captions:
                if random.random() < guidance_prob:
                    dropped_captions.append("")
                else:
                    dropped_captions.append(c)

            text_inputs = tokenizer(
                dropped_captions,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            ).to(device)
            emb = text_encoder(text_inputs.input_ids).last_hidden_state

            bsz = latents.shape[0]
            timesteps = torch.randint(
                0, scheduler.num_train_timesteps, (bsz,), device=device
            ).long()
            noise = torch.randn_like(latents)
            noisy_latents = scheduler.add_noise(latents, noise, timesteps)

            model_out = unet(noisy_latents, timesteps, encoder_hidden_states=emb)
            pred = model_out.sample
            loss = F.mse_loss(pred, noise)
            total_loss += loss.item()
            total_steps += 1

    unet.train()
    text_encoder.train()
    return total_loss / max(1, total_steps)


def train_conditional(
    data_root="datasets/colors",  # expects train/ and val/
    output_dir="color_diffusion_cond_checkpoints",
    epochs=10,
    batch_size=4,
    lr=1e-4,
    cfg_prob=0.1,  # fraction of prompts replaced with empty for training
    guidance_scale=7.5,  # used in sample_images_cfg
    val_guidance_prob=0.0,  # if you want to drop text in val
):
    """
    - cfg_prob: fraction of the time we drop the prompt (to learn unconditional).
    - guidance_scale: how strongly we weigh the prompt vs. no-prompt during sampling.
    """
    wandb.init(project="diffusion", entity="aicellio", name="color-diffusion-CFG")

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

    # Gather color folder names from train split (used for sampling prompts)
    available_colors = [
        d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))
    ]
    print(f"Train split colors: {available_colors}")

    # 1. Load VAE
    vae = AutoencoderKL.from_pretrained(
        "CompVis/stable-diffusion-v1-4", subfolder="vae"
    ).to(device)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)

    # 2. Load Text Encoder + Tokenizer
    tokenizer = CLIPTokenizer.from_pretrained(
        "CompVis/stable-diffusion-v1-4", subfolder="tokenizer"
    )
    text_encoder = CLIPTextModel.from_pretrained(
        "CompVis/stable-diffusion-v1-4", subfolder="text_encoder"
    ).to(device)
    text_encoder.eval()
    for p in text_encoder.parameters():
        p.requires_grad_(False)

    # 3. Create conditional UNet2DConditionModel
    unet = UNet2DConditionModel(
        sample_size=64,
        in_channels=4,
        out_channels=4,
        cross_attention_dim=768,
        layers_per_block=2,
    ).to(device)

    scheduler = DDPMScheduler(num_train_timesteps=1000)

    # 4. Datasets & Dataloaders
    train_dataset = ColorDatasetWithText(train_dir)
    val_dataset = ColorDatasetWithText(val_dir)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, drop_last=True
    )

    # 5. Optimizer
    optimizer = optim.AdamW(unet.parameters(), lr=lr)

    global_step = 0

    for epoch in range(epochs):
        unet.train()

        for step, (images, captions) in enumerate(train_loader):
            images = images.to(device)
            with torch.no_grad():
                latents = vae.encode(images).latent_dist.sample() * 0.18215

            # -------------------------------------------------
            # Classifier-Free Guidance during TRAINING
            # We randomly drop the prompt ~cfg_prob% of the time
            # -------------------------------------------------
            dropped_captions = []
            for c in captions:
                if random.random() < cfg_prob:
                    dropped_captions.append("")
                else:
                    dropped_captions.append(c)

            # Encode text
            text_inputs = tokenizer(
                dropped_captions,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            ).to(device)
            encoder_hidden_states = text_encoder(
                text_inputs.input_ids
            ).last_hidden_state

            bsz = latents.shape[0]
            timesteps = torch.randint(
                0, scheduler.num_train_timesteps, (bsz,), device=device
            ).long()
            noise = torch.randn_like(latents)
            noisy_latents = scheduler.add_noise(latents, noise, timesteps)

            model_out = unet(
                noisy_latents, timesteps, encoder_hidden_states=encoder_hidden_states
            )
            pred = model_out.sample

            loss = F.mse_loss(pred, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            wandb.log({"train/loss": loss.item(), "step": global_step})
            global_step += 1

            if step % 50 == 0:
                print(f"Epoch {epoch} | Step {step} | Loss {loss.item():.4f}")

        # ---------------------------
        # Validation
        # ---------------------------
        val_loss = validate_one_epoch(
            unet,
            vae,
            scheduler,
            text_encoder,
            tokenizer,
            val_loader,
            device,
            guidance_prob=val_guidance_prob,
        )
        wandb.log({"val/loss": val_loss, "epoch": epoch})
        print(f"Epoch {epoch} | Validation Loss: {val_loss:.4f}")

        # ---------------------------
        # Sample images with CFG
        # ---------------------------
        random_color = random.choice(available_colors)
        example_prompt = f"a photo of a {random_color} color"
        images_out, used_prompt = sample_images_cfg(
            unet,
            vae,
            scheduler,
            text_encoder,
            tokenizer,
            device,
            prompt=example_prompt,
            guidance_scale=guidance_scale,
            num_samples=4,
            num_inference_steps=30,
        )
        grid = make_grid(images_out, nrow=2)
        grid_pil = to_pil_image(grid)

        grid_path = os.path.join(output_dir, f"samples_epoch_{epoch}.png")
        grid_pil.save(grid_path)

        wandb.log(
            {
                "epoch": epoch,
                "sample_images": wandb.Image(
                    grid_pil, caption=f"Epoch {epoch} - Prompt: {used_prompt}"
                ),
            }
        )

        # Save checkpoint
        if epoch % 5 == 0:
            ckpt_path = os.path.join(output_dir, f"unet_epoch_{epoch}.pt")
            torch.save(unet.state_dict(), ckpt_path)
            print(f"Saved checkpoint to {ckpt_path}")

    ckpt_path = os.path.join(output_dir, "final_unet.pt")
    torch.save(unet.state_dict(), ckpt_path)
    print(f"Saved final checkpoint to {ckpt_path}")


if __name__ == "__main__":
    train_conditional()
