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
FALLBACKS: int =0
class Subtree(NamedTuple):
    root: int
    nodes: dict[int, dict[str, str]]
    edges: list[dict[str, Any]]

def module_count(genome: TreeGenome) -> int:
    return len(genome.nodes)




def reset_fallbacks() -> None:
    global FALLBACKS
    FALLBACKS = 0


def child_on_face(genome: TreeGenome, face: str) -> int | None:
    for edge in genome.edges:
        if edge["parent"] ==IDX_OF_CORE and edge["face"] == face:
            return edge["child"]
    return None



def subtree_on_face(genome: TreeGenome, face: str) -> Subtree | None:
    root =child_on_face(genome, face)
    if root is None or root not in genome.nodes:
        return None

    members ={root, *nx.descendants(genome.to_networkx(), root)}
    nodes= {
        node: copy.deepcopy(genome.nodes[node])
        for node in members
    }
    edges= [
        copy.deepcopy(edge)
        for edge in genome.edges
        if edge["parent"] in members and edge["child"] in members
    ]
    return Subtree(root, nodes, edges)



def graft(child: TreeGenome, subtree: Subtree, face: str, next_id: int) -> int:
    new_id={}
    for old_id in sorted(subtree.nodes):
        new_id[old_id]=next_id
        next_id +=1

    for old_id, attributes in subtree.nodes.items():
        child.nodes[new_id[old_id]]= attributes
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
    global FALLBACKS

    if not parents:
        msg ="recombine needs at least one parent"
        raise ValueError(msg)

    base= parents[0]
    child =TreeGenome(
        nodes={IDX_OF_CORE: copy.deepcopy(base.nodes[IDX_OF_CORE])},
        edges=[],
    )
    next_id=IDX_OF_CORE + 1

    for face in CORE_FACES:
        donor= parents[rng.randrange(len(parents))]
        subtree =subtree_on_face(donor, face)
        if subtree is not None:
            next_id =graft(child, subtree, face, next_id)

    try:
        validate_genome_dict(child.to_dict())
    except ValueError:
        FALLBACKS += 1
        return copy.deepcopy(base)
    return child
