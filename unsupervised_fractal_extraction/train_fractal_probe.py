import argparse
import datetime
import json
from functools import partial
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from epsilon_transformers.analysis.activation_analysis import \
    get_beliefs_for_transformer_inputs
from epsilon_transformers.process.processes import Mess3
from epsilon_transformers.training.configs.model_configs import RawModelConfig
from epsilon_transformers.training.configs.training_configs import \
    ProcessDatasetConfig
from torch import nn
from torch.optim import Adam, Optimizer
from tqdm import tqdm, trange
from transformer_lens.hook_points import HookPoint

from utils import (BoxCountingDimensionLoss, CorrelationDimensionLoss,
                   LinearProbe, LinearProbeWithSoftmax)


def main(args: argparse.Namespace):
    torch.set_default_device(args.device)

    # Load model
    checkpoint_path = Path(
        "/workspaces/cure/compmech-models/models/f6gnm1we-mess3-0.15-0.6/"
    )
    weights = torch.load(checkpoint_path / "998406400.pt")

    with open(checkpoint_path / "train_config.json", "r") as f:
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

    model = train_config.to_hooked_transformer(device=torch.device(args.device))
    model.load_state_dict(weights)
    print("Weights loaded successfully.")

    # Cache all HMM paths
    mess3 = Mess3()
    msp_tree = mess3.derive_mixed_state_presentation(depth=model.cfg.n_ctx + 1)
    tree_paths, tree_beliefs = msp_tree.paths_and_belief_states

    transformer_inputs = [x for x in tree_paths if len(x) == train_config.n_ctx]
    transformer_inputs = torch.tensor(transformer_inputs, dtype=torch.int).to(
        args.device
    )
    print("HMM paths cached.")

    # Train
    training_config_dict = yaml.safe_load(open(args.config, "r"))

    linear_probe = LinearProbe(64, 3).cuda()
    optimizer = Adam(linear_probe.parameters(), lr=training_config_dict["lr"])

    r_values = np.logspace(
        training_config_dict["logspace_min"],
        training_config_dict["logspace_max"],
        num=training_config_dict["logspace_num"],
    )

    loss_fn: nn.Module
    if training_config_dict["loss"] == "CDL":
        loss_fn = CorrelationDimensionLoss(r_values)
    elif training_config_dict["loss"] == "BCD":
        loss_fn = BoxCountingDimensionLoss(
            epsilons=r_values,
            sigma=training_config_dict["sigma"],
            spread_weight=training_config_dict["spread_weight"],
            p=training_config_dict["p"],
        )

    def probe_train_hook(attn_result: torch.Tensor, hook: HookPoint) -> torch.Tensor:
        optimizer.zero_grad()
        probe_out = linear_probe(attn_result)
        loss = loss_fn(probe_out)
        loss.backward()
        optimizer.step()
        return attn_result

    model.eval()
    transformer_input_loader = torch.utils.data.DataLoader(
        transformer_inputs, batch_size=training_config_dict["batch_size"]
    )

    for i in trange(training_config_dict["num_iters"]):
        for transformer_input in tqdm(transformer_input_loader):
            _ = model.run_with_hooks(
                transformer_input,
                fwd_hooks=[("blocks.3.hook_resid_post", probe_train_hook)],
            )

    def probe_log_hook(
        attn_result: torch.Tensor, hook: HookPoint, belief_guess_list: List
    ) -> torch.Tensor:
        probe_out = linear_probe(attn_result)
        belief_guess_list.append(probe_out)
        return attn_result

    belief_guess_list = []
    for transformer_input in transformer_input_loader:
        hook_with_beliefs = partial(probe_log_hook, belief_guess_list=belief_guess_list)
        _ = model.run_with_hooks(
            transformer_input,
            fwd_hooks=[("blocks.3.hook_resid_post", hook_with_beliefs)],
        )
        break
    print("Generated belief guesses using linear probe:")

    points = torch.cat(belief_guess_list)
    points = points.view(points.shape[0] * points.shape[1], points.shape[2])

    print(points[:5])

    # Function to convert ternary coordinates to Cartesian coordinates
    def ternary_to_cartesian(triples):
        points = []
        for triple in triples:
            x = 0.5 * (2 * triple[1] + triple[2])
            y = (np.sqrt(3) / 2) * triple[2]
            points.append([x, y])
        return np.array(points)

    # Convert ternary coordinates to Cartesian coordinates
    ternary_points = ternary_to_cartesian(points.cpu().detach().numpy())
    # Plotting
    fig, ax = plt.subplots()
    ax.set_aspect("equal")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, np.sqrt(3) / 2)

    # Draw the triangle
    triangle = plt.Polygon(
        [[0, 0], [1, 0], [0.5, np.sqrt(3) / 2]], edgecolor="k", fill=None
    )
    ax.add_patch(triangle)

    # Plot points
    ax.scatter(
        ternary_points[:, 0],
        ternary_points[:, 1],
        color="red",
        label="Sample Points",
        s=0.1,
    )

    # Show plot
    n = 5
    title = "Mess3 -- " + "\n".join(
        ", ".join(
            f"{k}: {training_config_dict[k]}"
            for k in list(training_config_dict.keys())[i : i + n]
        )
        for i in range(0, len(training_config_dict), n)
    )
    plt.title(title, fontsize=10)
    plt.legend()

    plt.savefig(args.image)
    torch.save({"probe": linear_probe, "config": training_config_dict}, args.probe)

    plt.close()
    print(f"Saved probe weights in {args.probe}, saved image in {args.image}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Configuration flags
    parser.add_argument("--device", default="cpu", type=str)
    parser.add_argument(
        "--config",
        default="unsupervised_fractal_extraction/train_fractal_probe_config.yaml",
        type=str,
    )

    # Output flags
    now = str(datetime.datetime.now()).replace(" ", "--").replace(":", "-")
    parser.add_argument(
        "--image",
        default="unsupervised_fractal_extraction/outputs/output_{}.jpg".format(now),
        type=str,
    )
    parser.add_argument(
        "--probe",
        default="unsupervised_fractal_extraction/models/model_{}.pt".format(now),
        type=str,
    )

    args = parser.parse_args()
    main(args)
