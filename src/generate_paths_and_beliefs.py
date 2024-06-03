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

from utils import get_cached_belief_filename


def derive_mixed_state_presentation(process: Process, depth: int) -> MixedStateTree:
    starting_prob_vec = np.expand_dims(process.steady_state_vector, axis=0)
    tree_root = MixedStateTreeNode(
        state_prob_vector=starting_prob_vec, children=set(), path=[], emission_prob=0
    )
    nodes = set([tree_root])

    stack = deque([(tree_root, starting_prob_vec, [], 0)])
    while stack:
        current_node, state_prob_vector, current_path, current_depth = stack.pop()
        # print(state_prob_vector.shape, current_path)
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

                    child_node = MixedStateTreeNode(
                        state_prob_vector=full_state_prob_history,
                        path=child_path,
                        children=set(),
                        emission_prob=emission_probs[emission],
                    )
                    current_node.add_child(child_node)

                    stack.append(
                        (
                            child_node,
                            full_state_prob_history,
                            child_path,
                            current_depth + 1,
                        )
                    )
        nodes.add(current_node)

    return MixedStateTree(
        root_node=tree_root, process=process.name, nodes=nodes, depth=depth
    )


def main(args: argparse.Namespace):
    with open(args.config, "r") as f:
        msp_cfg = yaml.safe_load(f)

    for x, a in tqdm(msp_cfg):
        assert isinstance(x, float)
        assert isinstance(a, float)

        mess3 = Mess3(x=x, a=a)
        msp_with_full_belief_history = derive_mixed_state_presentation(mess3, 10)

        tree_paths, tree_beliefs = msp_with_full_belief_history.paths_and_belief_states

        seq_len = 10
        pairs = [
            (path, beliefs)
            for path, beliefs in zip(tree_paths, tree_beliefs)
            if len(path) == seq_len
        ]
        inputs = torch.stack([torch.tensor(path) for (path, beliefs) in pairs])
        input_beliefs = torch.stack(
            [torch.tensor(beliefs) for (path, beliefs) in pairs]
        )

        file_path = get_cached_belief_filename(x, a)
        torch.save(
            {
                "params": {"x": x, "a": a},
                "inputs": inputs,
                "input_beliefs": input_beliefs,
            },
            file_path,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="src/msp_cfg.yaml", type=str)
    args = parser.parse_args()
    main(args)
