
"""
EP42 Challenge — Train a DCGAN on MNIST + diagnose mode collapse
pip install torch torchvision matplotlib
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
import matplotlib.pyplot as plt
import os

# ── Hyperparameters ──────────────────────────────────────────────────────────
LATENT_DIM = 100
BATCH_SIZE = 128
EPOCHS     = 50
LR         = 2e-4
BETA1      = 0.5
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DIR   = "dcgan_grids"
os.makedirs(SAVE_DIR, exist_ok=True)

# ── Data (normalise to [-1, 1] to match Tanh) ────────────────────────────────
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),   # [0,1] → [-1,1]
])
train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

# ── Generator ────────────────────────────────────────────────────────────────
class Generator(nn.Module):
    def __init__(self, latent_dim=100):
        super().__init__()
        self.main = nn.Sequential(
            # input: latent_dim x 1 x 1
            nn.ConvTranspose2d(latent_dim, 256, 7, 1, 0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # 256 x 7 x 7
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # 128 x 14 x 14
            nn.ConvTranspose2d(128, 1, 4, 2, 1, bias=False),
            nn.Tanh(),
            # 1 x 28 x 28
        )

    def forward(self, z):
        return self.main(z.view(z.size(0), -1, 1, 1))


# ── Discriminator ────────────────────────────────────────────────────────────
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            # input: 1 x 28 x 28
            nn.Conv2d(1, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 64 x 14 x 14
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # 128 x 7 x 7
            nn.Conv2d(128, 1, 7, 1, 0, bias=False),
            nn.Sigmoid(),
            # 1 x 1 x 1
        )

    def forward(self, x):
        return self.main(x).view(-1, 1).squeeze(1)


# ── Init + optimisers ────────────────────────────────────────────────────────
netG = Generator(LATENT_DIM).to(DEVICE)
netD = Discriminator().to(DEVICE)

# DCGAN weight init
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

netG.apply(weights_init)
netD.apply(weights_init)

criterion = nn.BCELoss()
optimG = optim.Adam(netG.parameters(), lr=LR, betas=(BETA1, 0.999))
optimD = optim.Adam(netD.parameters(), lr=LR, betas=(BETA1, 0.999))

# fixed noise for consistent grids
fixed_noise = torch.randn(64, LATENT_DIM, 1, 1, device=DEVICE)

# ── Training loop ────────────────────────────────────────────────────────────
print(f"Training on {DEVICE} …")
for epoch in range(1, EPOCHS + 1):
    for real_imgs, _ in train_loader:
        real_imgs = real_imgs.to(DEVICE)
        b = real_imgs.size(0)
        real_label = torch.ones(b, device=DEVICE)
        fake_label = torch.zeros(b, device=DEVICE)

        # ── Update Discriminator ─────────────────────────────────────────────
        netD.zero_grad()
        # real
        out_real = netD(real_imgs)
        loss_D_real = criterion(out_real, real_label)
        # fake
        noise = torch.randn(b, LATENT_DIM, 1, 1, device=DEVICE)
        fake_imgs = netG(noise)
        out_fake = netD(fake_imgs.detach())          # important: detach
        loss_D_fake = criterion(out_fake, fake_label)
        loss_D = loss_D_real + loss_D_fake
        loss_D.backward()
        optimD.step()

        # ── Update Generator (non-saturating) ────────────────────────────────
        netG.zero_grad()
        out_fake_for_G = netD(fake_imgs)             # no detach this time
        loss_G = criterion(out_fake_for_G, real_label)  # want D to say "real"
        loss_G.backward()
        optimG.step()

    # ── Save grid every 5 epochs ─────────────────────────────────────────────
    if epoch % 5 == 0 or epoch == 1:
        with torch.no_grad():
            fake = netG(fixed_noise).detach().cpu()
            grid = utils.make_grid(fake, nrow=8, normalize=True, value_range=(-1, 1))
            utils.save_image(grid, f"{SAVE_DIR}/epoch_{epoch:03d}.png")
            print(f"Epoch {epoch:03d}  loss_D={loss_D.item():.3f}  loss_G={loss_G.item():.3f}  → grid saved")

print("Training finished.")
print(f"Grids saved in ./{SAVE_DIR}/")
print("Look at the grids — do digits become recognisable?")
print("Do all 64 images look like the same digit? (mode collapse check)")
