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
```bash
python nmf.py --values values.csv --components 16 --iterations 1000 --plot

### VAE
```bash
python vae.py --values values.csv --row_info row_info.csv --col_info col_info.csv --output_png results.png
