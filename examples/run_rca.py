"""
===================================
Plotting Reliable Component Analysis
===================================

Example usage of the RCA class.
"""

import pathlib
import click
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from rca_fmri import RCA

sns.set_theme(style="darkgrid")


@click.command()
@click.option(
    "--data-dir",
    type=click.Path(exists=True, file_okay=False, path_type=pathlib.Path),
    required=True,
    help="Directory containing features.npy and labels.npy.",
)
@click.option("--lr", type=float, default=5e-3, show_default=True, help="Learning rate.")
@click.option("--epochs", type=int, default=200, show_default=True, help="Number of epochs.")
@click.option(
    "--dim",
    "n_components",
    type=int,
    default=5,
    show_default=True,
    help="Embedding dimension (number of components).",
)
@click.option(
    "--batch-size",
    "batch",
    type=int,
    default=200,
    show_default=True,
    help="Batch size. Using a large batch size is recommended.",
)
@click.option(
    "--loss-type",
    "loss",
    type=str,
    default="contrastive",
    show_default=True,
    help="Type of contrastive loss. Can be 'contrastive' or 'info_nce'",
)
@click.option(
    "--penalty-scale",
    "penalty_scale",
    type=float,
    default=0.1,
    show_default=True,
    help="Penalty scale to achieve de-correlation of successive embedding dimensions.",
)
@click.option(
    "--weight-decay",
    "weight_decay",
    type=float,
    default=1e-3,
    show_default=True,
    help="Weight decay of Adam optimiser.",
)
@click.option(
    "--out-dir",
    type=click.Path(file_okay=False, path_type=pathlib.Path),
    default=None,
    help="Optional output directory to save the trained model.",
)
@click.option(
    "--show-plots",
    "plots",
    type=bool,
    default=True,
    help="Show plots. If you do not want to display plots, set to False.",
)
def main(data_dir, lr, epochs, n_components, batch, loss, penalty_scale, weight_decay, out_dir, plots):
    # Load data
    features_path, labels_path = data_dir / "std.npy", data_dir / "labels.npy"
    assert features_path.exists() and labels_path.exists()
    f"Data directory ({data_dir.as_posix()}) must contain"
    X = np.load(features_path).squeeze()
    X = (X - np.mean(X)) / np.std(X)
    labels = np.load(labels_path)
    print(f"Loaded X: {X.shape}, labels: {labels.shape}")

    # Fit RCA and transform to embeddings
    rca = RCA(
        n_components=n_components,
        lr=lr,
        n_epochs=epochs,
        batch_size=batch,
        model_type="linear",
        penalty_scale=penalty_scale,
        weight_decay=weight_decay,
        loss_type=loss,
    )
    print(f"Fitting RCA with n_components={n_components}")
    rca.fit(X, labels)
    print("Finished fitting RCA.")
    embeddings = rca.transform(X)
    print(f"Computed embeddings: {embeddings.shape}")

    # Compute scores per embedding dimension
    scores = [rca.score(X, labels, dim=dim) for dim in range(n_components)]
    dims = np.arange(1, n_components + 1)

    if plots:
        # Plot score vs. embedding dimension
        print("Plotting score per embedding dimension...")
        plt.figure(figsize=(8, 5))
        plt.plot(dims, scores, marker="o")
        plt.xticks(dims)
        plt.xlabel("Embedding dimension")
        plt.ylabel("ICC(1,1) score")
        plt.title("RCA score per embedding dimension")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

        # Plot training losses per component
        losses = rca.losses_
        plt.figure(figsize=(8, 5))
        for idx in range(losses.shape[0]):
            plt.plot(losses[idx], label=f"Component {idx + 1}")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training loss per component")
        plt.yscale("log")
        plt.legend(frameon=False, ncol=2)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    else:
        print("Plots disabled; reporting scores and final losses.")
        score_pairs = ", ".join(f"{dim}:{score:.4f}" for dim, score in zip(dims, scores))
        print(f"Scores (dim: ICC): {score_pairs}")
        losses = rca.losses_
        if losses.size:
            final_losses = losses[:, -1]
            loss_pairs = ", ".join(f"{idx + 1}:{loss:.6g}" for idx, loss in enumerate(final_losses))
            print(f"Final training loss per component: {loss_pairs}")

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        model_path = out_dir / "rca.pkl"
        rca.save(model_path)
        print(f"Saved model to {model_path}")


if __name__ == "__main__":
    main()
