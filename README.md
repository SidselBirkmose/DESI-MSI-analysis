# DESI-MS Fingermark Separation using NMF and VAE

This repository contains Python implementations of non-negative matrix factorization (NMF)
and a variational autoencoder (VAE) applied to DESI-MS imaging data of overlapping fingermarks.

## Requirements
- Python ≥ 3.9
- numpy
- pandas
- matplotlib
- torch

Install dependencies:
pip install numpy pandas matplotlib torch

## Data
Due to privacy and forensic restrictions, raw DESI-MS data are not publicly available.
The code expects input data in CSV format:
- values.csv (intensity matrix)
- row_info.csv (pixel coordinates)
- col_info.csv (m/z values)

## Usage

### NMF
**Command-line arguments:**
```bash
python nmf.py \
--values VALUES \
[--row_info ROW_INFO] \
[--components COMPONENTS] \
[--iterations ITERATIONS] \
[--lr LR] \
[--save_output] \
[--plot] \
[--plot_cols PLOT_COLS] \
[--plot_output PLOT_OUTPUT] \
[--flip_y] \
[--flip_x] 
```
--values (required): CSV file containing the data matrix \n
--row_info (optional, required for plotting): CSV with row metadata (coordinates)
--components: Number of NMF components (default: 16)
--iterations: Training iterations (default: 1000)
--lr: Learning rate (default: 0.01)
--save_output: Save factorization matrices W.csv and H.csv
--plot:  Plot NMF components
--plot_cols: Number of columns in plot grid (default: 4)
--plot_output: Filename for component plot (default: nmf_components.png)
--flip_y/--flip_x: Flip axes for visualization 

Results in Figure 6 and 7 were obtained by running the following:
```bash
python nmf.py \
--values values.csv \
--row_info row_info.csv \
--components 16 \
--iterations 1000 \
--plot
```


### VAE
```bash
python vae.py \
--row_info ROW_INFO \
--col_info COL_INFO \
--values VALUES \
[--latent_dim LATENT_DIM] \
[--epochs EPOCHS] \
[--lr LR] \
[--nrows NROWS] \
--output_png OUTPUT_PNG
```

Results in Figure 8 were obtained by running the following:
```bash
python vae.py \
    --row_info row_info.csv \
    --col_info col_info.csv \
    --values values.csv \
    --latent_dim 16 \
    --epochs 100 \
    --output_png VAE_100_epochs.png
```

