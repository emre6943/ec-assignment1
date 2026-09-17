"""Face-aligned multi-parent recombination for ARIEL tree genomes.

Block B of the team task list owns this file. The version here was written
alongside the EA so the pipeline runs end to end; the internals may be
replaced, but ``ea.py`` and the tests depend on this public surface:

    recombine(parents: list[TreeGenome], rng: random.Random) -> TreeGenome
    module_count(genome: TreeGenome) -> int
    FALLBACKS: int

The child starts as a copy of ``parents[0]``'s core. For each core face a donor
is drawn uniformly from the parents, and whatever subtree hangs on that face of
that donor is deep-copied onto the same face of the child with fresh ids.

With k = 2 this is the control condition. It deliberately does not use ARIEL's
``crossover_subtree``, so that the number of parents stays the only difference
between experiment variants.

Exactly one ``rng`` draw happens per face, in ``CORE_FACES`` order, and nothing
else consumes the generator, so a caller who knows the seed can predict which
parent supplied which face. The unit tests rely on that.
"""

from __future__ import annotations

import copy
import random
from typing import Any, NamedTuple

import networkx as nx

from ariel.body_phenotypes.robogen_lite.config import (
    ALLOWED_FACES,
    IDX_OF_CORE,
    ModuleType,
)
from ariel.ec.genotypes.tree.tree_genome import TreeGenome
from ariel.ec.genotypes.tree.validation import validate_genome_dict

CORE_FACES: tuple[str, ...] = tuple(
    face.name for face in ALLOWED_FACES[ModuleType.CORE]
)

FALLBACKS: int = 0


class Subtree(NamedTuple):
    root: int
    nodes: dict[int, dict[str, str]]
    edges: list[dict[str, Any]]


def module_count(genome: TreeGenome) -> int:
    """Number of modules in a genome, core included."""
    return len(genome.nodes)


def reset_fallbacks() -> None:
    global FALLBACKS
    FALLBACKS = 0


def child_on_face(genome: TreeGenome, face: str) -> int | None:
    """Id of the module attached to that face of the core, if any."""
    for edge in genome.edges:
        if edge["parent"] == IDX_OF_CORE and edge["face"] == face:
            return edge["child"]
    return None


def subtree_on_face(genome: TreeGenome, face: str) -> Subtree | None:
    """Deep-copy the core subtree on ``face``, still carrying the donor's ids."""
    root = child_on_face(genome, face)
    if root is None or root not in genome.nodes:
        return None

    members = {root, *nx.descendants(genome.to_networkx(), root)}
    nodes = {
        node: copy.deepcopy(genome.nodes[node])
        for node in members
    }
    edges = [
        copy.deepcopy(edge)
        for edge in genome.edges
        if edge["parent"] in members and edge["child"] in members
    ]
    return Subtree(root, nodes, edges)


def graft(child: TreeGenome, subtree: Subtree, face: str, next_id: int) -> int:
    """Attach ``subtree`` to ``face`` of the child's core, renumbering from
    ``next_id``. Returns the first id still free afterwards."""
    new_id = {}
    for old_id in sorted(subtree.nodes):
        new_id[old_id] = next_id
        next_id += 1

    for old_id, attributes in subtree.nodes.items():
        child.nodes[new_id[old_id]] = attributes
    for edge in subtree.edges:
        child.edges.append({
            "parent": new_id[edge["parent"]],
            "child": new_id[edge["child"]],
            "face": edge["face"],
        })
    child.edges.append({
        "parent": IDX_OF_CORE,
        "child": new_id[subtree.root],
        "face": face,
    })
    return next_id


def recombine(parents: list[TreeGenome], rng: random.Random) -> TreeGenome:
    """Build one child from ``parents``, sharing no mutable state with them.

    Falls back to a copy of ``parents[0]`` (and counts it in ``FALLBACKS``) if
    the assembled child does not validate.
    """
    global FALLBACKS

    if not parents:
        msg = "recombine needs at least one parent"
        raise ValueError(msg)

    base = parents[0]
    child = TreeGenome(
        nodes={IDX_OF_CORE: copy.deepcopy(base.nodes[IDX_OF_CORE])},
        edges=[],
    )
    next_id = IDX_OF_CORE + 1

    for face in CORE_FACES:
        donor = parents[rng.randrange(len(parents))]
        subtree = subtree_on_face(donor, face)
        if subtree is not None:
            next_id = graft(child, subtree, face, next_id)

    try:
        validate_genome_dict(child.to_dict())
    except ValueError:
        FALLBACKS += 1
        return copy.deepcopy(base)
    return child
