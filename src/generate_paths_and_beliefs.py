import argparse
from collections import deque

import numpy as np
import torch
import yaml
from epsilon_transformers.process.MixedStateTree import (MixedStateTree,
                                                         MixedStateTreeNode)
from epsilon_transformers.process.Process import (
    Process, _compute_emission_probabilities, _compute_next_distribution)
from epsilon_transformers.process.processes import Mess3
from tqdm import tqdm

from src.utils import get_cached_belief_filename
from typing import Tuple, Set, List


def generate_beliefs_for_depth(process: Process, depth: int) -> MixedStateTree:
    starting_prob_vec = np.expand_dims(process.steady_state_vector, axis=0)

    paths: List[Tuple[List[int], np.ndarray]] = []

    stack = deque([(starting_prob_vec, [], 0)])
    while stack:
        state_prob_vector, current_path, current_depth = stack.pop()
        if current_depth < depth:
            emission_probs = _compute_emission_probabilities(
                process, state_prob_vector[-1]
            )
            for emission in range(process.vocab_len):
                if emission_probs[emission] > 0:
                    next_state_prob_vector = np.expand_dims(
                        _compute_next_distribution(
                            process.transition_matrix, state_prob_vector[-1], emission
                        ),
                        axis=0,
                    )
                    child_path = current_path + [emission]

                    full_state_prob_history = (
                        np.concatenate(
                            [state_prob_vector, next_state_prob_vector], axis=0
                        )
                        if current_depth > 0
                        else next_state_prob_vector
                    )

                    if current_depth == depth - 1:
                        paths.append((child_path, full_state_prob_history))

                    stack.append(
                        (
                            full_state_prob_history,
                            child_path,
                            current_depth + 1,
                        )
                    )

    return paths


def generate_mess3_beliefs(x: float, a: float, sort_pairs: bool = False):
    mess3 = Mess3(x=x, a=a)
    seq_len = 10
    pairs = generate_beliefs_for_depth(mess3, seq_len)
    
    if sort_pairs:
        pairs = sorted(pairs, key=lambda x: x[0])

    inputs = torch.stack([torch.tensor(path) for (path, beliefs) in pairs])
    input_beliefs = torch.stack(
        [torch.tensor(beliefs) for (path, beliefs) in pairs]
    )

    return inputs, input_beliefs


def save_beliefs(inputs: torch.Tensor, input_beliefs: torch.Tensor, x: float, a: float) -> None:
    file_path = get_cached_belief_filename(x, a)
    torch.save(
        {
            "params": {"x": x, "a": a},
            "inputs": inputs,
            "input_beliefs": input_beliefs,
        },
        file_path,
    )

def main(args: argparse.Namespace):
    with open(args.config, "r") as f:
        msp_cfg = yaml.safe_load(f)

    for x, a in tqdm(msp_cfg):
        assert isinstance(x, float)
        assert isinstance(a, float)

        inputs, input_beliefs = generate_mess3_beliefs(x, a, sort_pairs=True)
        save_beliefs(inputs, input_beliefs, x, a)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="src/msp_cfg.yaml", type=str)
    args = parser.parse_args()
    main(args)
