import os
import argparse
import time
import math
import pandas as pd
import torch
import matplotlib.pyplot as plt

def load_values(values_path):
    values_df = pd.read_csv(values_path, header=None, low_memory=False)

    values_df = values_df.iloc[1:].reset_index(drop=True)
    values_numeric = values_df.apply(pd.to_numeric, errors="coerce")

    if values_numeric.isna().any().any():
        raise ValueError("Non-numeric values found in values.csv")
    
    X = torch.tensor(values_numeric.to_numpy(), dtype=torch.float32)

    return X
    

def nmf(X, n_components, n_iterations, lr):
    n_samples, n_features = X.shape

    # Initialize W and H
    W = torch.rand((n_samples, n_components), requires_grad=True)
    H = torch.rand((n_components, n_features), requires_grad=True)

    optimizer = torch.optim.Adam([W, H], lr=lr)
    loss_fn = torch.nn.MSELoss()

    print("Starting NMF training...")
    for epoch in range(n_iterations):
        optimizer.zero_grad()

        X_hat = W @ H
        loss = loss_fn(X_hat, X)

        loss.backward()
        optimizer.step()

        # Enforce non-negativity
        with torch.no_grad():
            W.clamp_(min=0)
            H.clamp_(min=0)

        if epoch % max(1, n_iterations // 10) == 0:
            print(f"Epoch {epoch:5d} | Loss: {loss.item():.6f}")

    return W.detach(), H.detach()


def save_outputs(W, H):
    pd.DataFrame(W.numpy()).to_csv("W.csv", index=False)
    pd.DataFrame(H.numpy()).to_csv("H.csv", index=False)

    print("Saved W.csv, H.csv")


def plot_components(
    W,
    row_info,
    output_path="nmf_components.png",
    ncols=4,
    flip_y=False,
    flip_x=False,
):
    
    # Convert W to NumPy
    W_np = W.detach().cpu().numpy()

    # Optional flips
    row_info = row_info.copy()
    if flip_y:
        row_info["y"] = -row_info["y"]
    if flip_x:
        row_info["x"] = -row_info["x"]   # <-- flip x values

    n_components = W_np.shape[1]
    nrows = math.ceil(n_components / ncols)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(ncols * 1.5, nrows * 1.2),
        constrained_layout=True,
    )

    axes = axes.flat if n_components > 1 else [axes]

    for i in range(n_components):
        df = row_info.copy()
        df["intensity"] = W_np[:, i]

        # Pivot to 2D image grid
        image = df.pivot_table(
            index="y",
            columns="x",
            values="intensity",
            aggfunc="mean",
        )

        ax = axes[i]
        ax.imshow(
            image,
            cmap="cividis",
            origin="lower",
            aspect="equal",
        )

        ax.set_title(f"Component {i + 1}", fontsize=10)
        ax.axis("off")

    # Hide unused axes
    for j in range(n_components, nrows * ncols):
        axes[j].axis("off")

    plt.savefig(output_path, dpi=600)
    plt.close()

    print(f"Saved component plot to {output_path}")



def main():
    n_threads = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
    torch.set_num_threads(n_threads)
    print(f"Using {n_threads} CPU threads for PyTorch")


    parser = argparse.ArgumentParser(description="Gradient-based NMF with plotting")

    parser.add_argument("--values", required=True, help="values.csv")
    parser.add_argument("--row_info", help="row_info.csv (required only if --plot is used)")

    parser.add_argument("--components", type=int, default=16, help="Number of NMF components")
    parser.add_argument("--iterations", type=int, default=1000, help="Training iterations")
    parser.add_argument("--lr", type=float, default=1e-2, help="Learning rate")
    parser.add_argument("--save_output", action="store_true", help="Save factorization matrices")

    parser.add_argument("--plot", action="store_true", help="Plot NMF components")
    parser.add_argument("--plot_cols", type=int, default=4, help="Number of columns in component grid")
    parser.add_argument("--plot_output", default="nmf_components.png", help="Output plot filename")
    parser.add_argument("--flip_y", action="store_true", help="Flip y-axis for plotting")
    parser.add_argument("--flip_x", action="store_true", help="Flip x-axis for plotting")

    args = parser.parse_args()
    
    
    
	  # Checking if rowinfo is supplied if --plot. 
    if args.plot and not args.row_info:
        parser.error("--row_info is required when using --plot")
    
    
    start_time = time.time()  # <--- start timer
		
		# Load data
    X = load_values(args.values)

    if args.plot:
        row_info = pd.read_csv(args.row_info)
        if X.shape[0] != len(row_info):
	          raise ValueError(
                "Row count mismatch between values matrix and row_info:\n"
                f"  values rows   : {X.shape[0]}\n"
                f"  row_info rows : {len(row_info)}"
            )

   	# Running NMF
    W, H = nmf(
        X,
        n_components=args.components,
        n_iterations=args.iterations,
        lr=args.lr,
    )
    
    if args.save_output:
        save_outputs(W, H)
    
    
    if args.plot:
        plot_components(
            W,
            row_info=row_info,
            output_path=args.plot_output,
            ncols=args.plot_cols,
            flip_y=args.flip_y,
            flip_x=args.flip_x,
        )
        
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"\nTotal runtime: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")


if __name__ == "__main__":
    main()
