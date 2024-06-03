import argparse
import json
from collections import deque
from pathlib import Path
import pickle

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import yaml
from epsilon_transformers.analysis.activation_analysis import \
    find_msp_subspace_in_residual_stream
from epsilon_transformers.process.MixedStateTree import (MixedStateTree,
                                                         MixedStateTreeNode)
from epsilon_transformers.process.Process import (
    Process, _compute_emission_probabilities, _compute_next_distribution)
from epsilon_transformers.process.processes import Mess3
from epsilon_transformers.training.configs.model_configs import RawModelConfig
from epsilon_transformers.training.configs.training_configs import \
    ProcessDatasetConfig
from epsilon_transformers.visualization.plots import (
    _project_to_simplex, plot_ground_truth_and_evaluated_2d_simplex)
from sklearn.linear_model import LinearRegression
from tqdm import tqdm

from src.utils import (MODEL_PATH_005_085, MODEL_PATH_015_06,
                   get_cached_belief_filename, get_jpg_filename)

def run_activation_to_beliefs_regression(activations, ground_truth_beliefs):

    # make sure the first two dimensions are the same
    assert activations.shape[0] == ground_truth_beliefs.shape[0]
    assert activations.shape[1] == ground_truth_beliefs.shape[1]

    # flatten the activations
    batch_size, n_ctx, d_model = activations.shape
    belief_dim = ground_truth_beliefs.shape[-1]
    activations_flattened = activations.reshape(-1, d_model)  # [batch * n_ctx, d_model]
    ground_truth_beliefs_flattened = ground_truth_beliefs.reshape(
        -1, belief_dim
    )  # [batch * n_ctx, belief_dim]

    # run the regression
    regression = LinearRegression()
    regression.fit(activations_flattened, ground_truth_beliefs_flattened)

    # get the belief predictions
    belief_predictions = regression.predict(
        activations_flattened
    )  # [batch * n_ctx, belief_dim]
    belief_predictions = belief_predictions.reshape(batch_size, n_ctx, belief_dim)

    return regression, belief_predictions


def r_squared(y: torch.Tensor, y_hat: torch.Tensor):
    y_mean = torch.mean(y)

    # Calculate total sum of squares (SS_total)
    ss_total = torch.sum((y - y_mean) ** 2)

    # Calculate residual sum of squares (SS_residual)
    ss_residual = torch.sum((y - y_hat) ** 2)

    # Calculate R-squared
    r_squared = 1 - (ss_residual / ss_total)
    
    return r_squared


def normalized_rmse(y, y_hat):
    """
    Calculate the normalized RMSE for each regression using standard deviation and aggregate them.
    
    Parameters:
    y_true (torch.Tensor): True values, shape (batch_size, num_regs)
    y_pred (torch.Tensor): Predicted values, shape (batch_size, num_regs)
    
    Returns:
    torch.Tensor: Aggregated normalized RMSE.
    """
    # Calculate RMSE for each regression
    mse = torch.mean((y - y_hat) ** 2, dim=0)
    rmse = torch.sqrt(mse)
    
    # Calculate the standard deviation of each regression
    y_std = torch.std(y, dim=0)
    
    # Normalize RMSE by the standard deviation
    nrmse = rmse / y_std
    
    # Aggregate normalized RMSEs (mean or sum)
    aggregated_nrmse = torch.mean(nrmse)
    
    return aggregated_nrmse


def load_model(path_to_weights: Path, path_to_config: Path, device: torch.device):
    weights = torch.load(path_to_weights)

    with open(path_to_config, "r") as f:
        required_fields = [
            "d_vocab",
            "d_model",
            "n_ctx",
            "d_head",
            "n_heads",
            "n_layers",
        ]
        cfg_dict = {k: v for k, v in json.load(f).items() if k in required_fields}
        cfg_dict["n_head"] = cfg_dict.pop("n_heads")
        cfg_dict["d_mlp"] = 4 * cfg_dict["d_model"]
        train_config = RawModelConfig(**cfg_dict)

    model = train_config.to_hooked_transformer(device=device)
    model.load_state_dict(weights)

    return model


def main(args: argparse.Namespace):
    # TODO: expand to 0.15/0.85 model
    models = [
        (MODEL_PATH_015_06 / "998406400.pt", MODEL_PATH_015_06 / "train_config.json"),
        (MODEL_PATH_005_085 / "684806400.pt", MODEL_PATH_015_06 / "train_config.json"),
    ]
    device = torch.device(args.device)

    with open(args.config, "r") as f:
        msp_cfg = yaml.safe_load(f)

    r2_values = {}

    # Unchanged code for training probe + generating graphic
    for model_path, config_path in models:
        model = load_model(model_path, config_path, device)

        model_key = "-".join(str(model_path).split("-")[-2:]) # Key is of the form "0.15-0.6"
        r2_values[model_key] = {} 
        for x, a in msp_cfg:
            print(f"Evaluating: (x={x}, a={a}) for {model_path}")

            belief_dict = torch.load(get_cached_belief_filename(x, a))
            inputs = belief_dict["inputs"]
            input_beliefs = belief_dict["input_beliefs"]

            _, activations = model.run_with_cache(
                inputs, names_filter=lambda x: "resid_post" in x
            )
            acts = activations["blocks.3.hook_resid_post"].cpu().detach().numpy()
            regression, belief_predictions = run_activation_to_beliefs_regression(
                acts, input_beliefs
            )
            
            r2 = r_squared(input_beliefs, belief_predictions)
            r2_values[model_key][(x, a)] = r2
            print(f"- R2: {r2}")
            print(f"- MSE: {torch.mean(torch.square(torch.tensor(belief_predictions) - input_beliefs))}")

            # TODO: any way to make image gen faster?
            if not args.no_image:
                belief_predictions_flattened = belief_predictions.reshape(-1, 3)
                transformer_input_belief_flattened = input_beliefs.reshape(-1, 3)

                # project to simplex
                belief_true_projected = _project_to_simplex(transformer_input_belief_flattened)
                belief_pred_projected = _project_to_simplex(belief_predictions_flattened)

                rgb_colors = transformer_input_belief_flattened

                sns.set_context("paper")
                fig, axes = plt.subplots(1, 2, figsize=(5, 3))

                # Plotting the true beliefs projected onto the simplex
                axes[0].scatter(
                    belief_true_projected[0],
                    belief_true_projected[1],
                    marker=".",
                    c=rgb_colors,
                    alpha=0.2,
                    s=0.5,
                )
                axes[0].axis("off")
                axes[0].set_title("Ground Truth Simplex")

                # Plotting the predicted beliefs projected onto the simplex
                axes[1].scatter(
                    belief_pred_projected[0],
                    belief_pred_projected[1],
                    marker=".",
                    c=rgb_colors,
                    alpha=0.5,
                    s=0.01,
                )
                axes[1].axis("off")
                axes[1].set_title("Residual Stream Simplex")

                # Adjust layout for better spacing
                plt.tight_layout()

                # Display the figure
                plt.savefig(get_jpg_filename(model_path, x, a))

    with open("r2.pkl", "wb") as f:
        pickle.dump(r2_values, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="src/msp_cfg.yaml", type=str)
    parser.add_argument("--no-image", action="store_true")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    main(args)
