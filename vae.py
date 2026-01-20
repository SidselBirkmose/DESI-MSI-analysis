#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-GUI backend
import matplotlib.pyplot as plt
import argparse
import sys

# ------------------------------------------------------
# 1. Command-line arguments
# ------------------------------------------------------
parser = argparse.ArgumentParser(description="Run VAE on spectral data.")
parser.add_argument("--row_info", required=True, help="Path to row_info CSV file.")
parser.add_argument("--col_info", required=True, help="Path to col_info CSV file.")
parser.add_argument("--values", required=True, help="Path to values CSV file.")
parser.add_argument("--latent_dim", type=int, default=16, help="Latent space dimension.")
parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
parser.add_argument("--nrows", type=int, default=None, help="Subset number of rows for testing.")
parser.add_argument("--output_png", required=True, help="Filename for output PNG image.")
args = parser.parse_args()

# ------------------------------------------------------
# 2. Load Data
# ------------------------------------------------------
print(f"\n🔹 Loading data...", file=sys.stderr, flush=True)
row_info = pd.read_csv(args.row_info)
col_info = pd.read_csv(args.col_info)
values = pd.read_csv(args.values).to_numpy().astype(np.float32)

# Optional subset for testing
if args.nrows is not None:
    rng = np.random.default_rng(seed=42)  # reproducible
    idx = rng.choice(values.shape[0], size=args.nrows, replace=False)
    values = values[idx, :]
    row_info = row_info.iloc[idx].copy()

print(f"Data shape (subset for test): {values.shape}", file=sys.stderr, flush=True)

X = torch.tensor(values, dtype=torch.float32)
X = torch.clamp(X, min=0.0)  # ensure non-negativity

n_samples, n_features = X.shape

# ------------------------------------------------------
# 3. Define VAE
# ------------------------------------------------------
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        self.mu = nn.Linear(256, latent_dim)
        self.logvar = nn.Linear(256, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim),
            nn.Softplus()  # non-negative output
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.mu(h), self.logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

# ------------------------------------------------------
# 4. Loss function
# ------------------------------------------------------
def vae_loss(recon_x, x, mu, logvar):
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return (recon_loss + kl_loss) / x.size(0)

# ------------------------------------------------------
# 5. Device and model
# ------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}", file=sys.stderr, flush=True)

latent_dim = args.latent_dim
vae = VAE(n_features, latent_dim).to(device)
optimizer = optim.Adam(vae.parameters(), lr=args.lr)

X = X.to(device)

# ------------------------------------------------------
# 6. Training Loop
# ------------------------------------------------------
print("Starting training...", flush=True)
for epoch in range(1, args.epochs + 1):
    vae.train()
    optimizer.zero_grad()
    recon, mu, logvar = vae(X)
    loss = vae_loss(recon, X, mu, logvar)
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0 or epoch == 1:
        print(f"Epoch {epoch}/{args.epochs}, Avg Loss: {loss.item():.4f}", flush=True)

print("Training completed. Starting visualization...", flush=True)

# ------------------------------------------------------
# 7. Extract latent representations
# ------------------------------------------------------
vae.eval()
with torch.no_grad():
    mu, _ = vae.encode(X)
    Z = mu.cpu().numpy()  # samples x latent_dim

# ------------------------------------------------------
# 8. Visualization
# ------------------------------------------------------
# Drop pixel index if exists
row_info = row_info.drop(columns=["index"], errors="ignore")
row_info["y_flipped"] = -row_info["y"]

n_components = Z.shape[1]
ncols = 4
nrows = (n_components + ncols - 1) // ncols

fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*1.5, nrows*1.2), constrained_layout=True)
axes = axes.flatten()

for i in range(n_components):
    df = row_info.copy()
    df["intensity"] = Z[:, i]
    image = df.pivot_table(index="y", columns="x", values="intensity")

    ax = axes[i]
    im = ax.imshow(image, cmap="cividis", origin="lower", aspect="equal")
    ax.set_title(f"Latent {i+1}", fontsize=10)
    ax.axis("off")

for j in range(n_components, len(axes)):
    axes[j].axis("off")

plt.show()
print("✅ Done! Latent components visualized.", flush=True)

plt.savefig(args.output_png, dpi=300)
print(f"Saved figure to {args.output_png}")

