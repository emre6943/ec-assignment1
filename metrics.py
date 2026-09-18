"""Shared metrics used by EA, baseline and evaluation."""

import networkx as nx
import numpy as np

from tree_edit_distance import tree_edit_distance

def mean_pairwise_tree_distance(bodies: list[nx.DiGraph]) -> float:
    distances= [
        tree_edit_distance(one, other)
        for index, one in enumerate(bodies)
        for other in bodies[index + 1 :]
    ]
    return float(np.mean(distances)) if distances else 0.0
