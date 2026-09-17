"""Unit tests for the face-aligned multi-parent recombination operator."""

import copy
import random

import pytest

from ariel.body_phenotypes.robogen_lite.config import IDX_OF_CORE
from ariel.ec.genotypes.tree.tree_genome import TreeGenome
from ariel.ec.genotypes.tree.validation import validate_genome_dict

import recombine as recombine_mod
from recombine import CORE_FACES, module_count, recombine, subtree_on_face


def build(
    nodes: dict[int, tuple[str, str]],
    edges: list[tuple[int, int, str]],
) -> TreeGenome:
    g = TreeGenome()
    g.nodes = {IDX_OF_CORE: {"type": "CORE", "rotation": "DEG_0"}}
    for nid, (t, r) in nodes.items():
        g.nodes[nid] = {"type": t, "rotation": r}
    g.edges = [{"parent": p, "child": c, "face": f} for p, c, f in edges]
    validate_genome_dict(g.to_dict())
    return g


@pytest.fixture
def parents() -> list[TreeGenome]:
    # P0: FRONT -> BRICK -> (FRONT) HINGE ; BACK -> HINGE
    p0 = build(
        {1: ("BRICK", "DEG_0"), 2: ("HINGE", "DEG_45"), 3: ("HINGE", "DEG_0")},
        [(0, 1, "FRONT"), (1, 2, "FRONT"), (0, 3, "BACK")],
    )
    # P1: LEFT -> BRICK -> (TOP) HINGE ; FRONT -> HINGE
    p1 = build(
        {1: ("BRICK", "DEG_90"), 2: ("HINGE", "DEG_0"), 3: ("HINGE", "DEG_90")},
        [(0, 1, "LEFT"), (1, 2, "TOP"), (0, 3, "FRONT")],
    )
    # P2: RIGHT -> HINGE ; BACK -> BRICK -> (LEFT) BRICK -> (FRONT) HINGE
    p2 = build(
        {
            1: ("HINGE", "DEG_0"),
            2: ("BRICK", "DEG_45"),
            3: ("BRICK", "DEG_0"),
            4: ("HINGE", "DEG_90"),
        },
        [(0, 1, "RIGHT"), (0, 2, "BACK"), (2, 3, "LEFT"), (3, 4, "FRONT")],
    )
    return [p0, p1, p2]


def shape(genome: TreeGenome, face: str) -> tuple | None:
    """Id-free canonical description of the subtree on a core face."""
    picked = subtree_on_face(genome, face)
    if picked is None:
        return None
    root, nodes, edges = picked
    children = {}
    for e in edges:
        children.setdefault(e["parent"], []).append((e["face"], e["child"]))

    def walk(n: int) -> tuple:
        kids = tuple(
            (f, walk(c)) for f, c in sorted(children.get(n, []), key=lambda x: x[0])
        )
        return (nodes[n]["type"], nodes[n]["rotation"], kids)

    return walk(root)


def test_child_faces_come_from_expected_donors(parents):
    seed = 7
    expected_rng = random.Random(seed)
    expected_donor = {f: expected_rng.randrange(len(parents)) for f in CORE_FACES}

    child = recombine(parents, random.Random(seed))

    for face in CORE_FACES:
        donor = parents[expected_donor[face]]
        assert shape(child, face) == shape(donor, face), face


def test_every_face_matches_some_parent_over_many_seeds(parents):
    for seed in range(50):
        child = recombine(parents, random.Random(seed))
        for face in CORE_FACES:
            assert shape(child, face) in {shape(p, face) for p in parents}


def test_ids_unique_and_consecutive(parents):
    for seed in range(50):
        child = recombine(parents, random.Random(seed))
        ids = sorted(child.nodes)
        assert ids == list(range(len(ids)))
        assert ids[0] == IDX_OF_CORE
        # every non-core node has exactly one incoming edge
        incoming = {}
        for e in child.edges:
            assert e["parent"] in child.nodes and e["child"] in child.nodes
            incoming[e["child"]] = incoming.get(e["child"], 0) + 1
        for nid in ids[1:]:
            assert incoming.get(nid) == 1


def test_module_count_bounded_by_donors_and_valid(parents):
    total = sum(module_count(p) for p in parents)
    for seed in range(50):
        child = recombine(parents, random.Random(seed))
        assert 1 <= module_count(child) <= total
        validate_genome_dict(child.to_dict())


def test_child_shares_no_state_with_parents(parents):
    snapshot = [copy.deepcopy(p) for p in parents]
    child = recombine(parents, random.Random(3))
    child.edges.append({"parent": 0, "child": 999, "face": "TOP"})
    for n in child.nodes.values():
        n["rotation"] = "DEG_90"
    assert [p.to_dict() for p in parents] == [s.to_dict() for s in snapshot]


def test_core_copied_from_parent_zero(parents):
    child = recombine(parents, random.Random(0))
    assert child.nodes[IDX_OF_CORE] == parents[0].nodes[IDX_OF_CORE]


def test_deterministic_for_same_seed(parents):
    a = recombine(parents, random.Random(11))
    b = recombine(parents, random.Random(11))
    assert a.to_dict() == b.to_dict()


def test_fallback_on_invalid_child(parents, monkeypatch):
    def boom(_genome_dict):
        raise ValueError("forced")

    monkeypatch.setattr(recombine_mod, "validate_genome_dict", boom)
    recombine_mod.reset_fallbacks()
    child = recombine(parents, random.Random(0))
    assert recombine_mod.FALLBACKS == 1
    assert child.to_dict() == parents[0].to_dict()
    assert child is not parents[0]
    assert child.edges is not parents[0].edges
