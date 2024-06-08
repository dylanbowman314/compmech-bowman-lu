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
import itertools
import time


def generate_beliefs_for_depth(process: Process, depth: int) -> MixedStateTree:
    """
    Deprecated.
    """
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


def multiply_beliefs(batch1: torch.Tensor, batch2: torch.Tensor):
    accumulated_tm = torch.einsum('bij,cjk->bcik', batch1, batch2)
    num_symbols, num_seqs, n, _ = accumulated_tm.shape
    accumulated_tm = accumulated_tm.reshape(num_symbols * num_seqs, n, n)
    return accumulated_tm


def compute_all_beliefs(process: Process, seq_len: int, device=torch.device("cpu")):
    tm = torch.tensor(process.transition_matrix, dtype=torch.float32, device=device)
    accumulated_tm = torch.eye(len(tm), dtype=torch.float32, device=device).unsqueeze(0)

    for i in range(seq_len):
        accumulated_tm = multiply_beliefs(tm, accumulated_tm)

    return accumulated_tm


def compute_all_beliefs_with_history(process: Process, seq_len: int, device=torch.device("cpu")):
    tm = torch.tensor(process.transition_matrix, dtype=torch.float32, device=device)
    accumulated_tm = torch.eye(len(tm), dtype=torch.float32, device=device).unsqueeze(0)

    layers = []
    for i in range(seq_len):
        accumulated_tm = multiply_beliefs(tm, accumulated_tm)
        layers.append(accumulated_tm.repeat_interleave((tm.shape[-1])**(seq_len - i - 1), dim=0))

    return layers


def generate_mess3_beliefs(x: float, a: float, sort_pairs: bool = False):
    st = time.time()
    mess3 = Mess3(x=x, a=a)
    seq_len = 10

    prior = torch.tensor([1/3, 1/3, 1/3], device="cpu")
    input_beliefs = prior @ torch.stack(compute_all_beliefs_with_history(mess3, seq_len)).permute(1,0,2,3)
    input_beliefs = input_beliefs / torch.sum(input_beliefs, dim=2).unsqueeze(2)
    
    inputs = torch.tensor(list(itertools.product([0, 1, 2], repeat=10)), dtype=torch.int)

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
