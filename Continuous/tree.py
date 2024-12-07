from collections import defaultdict
import sys
import os
from typing import Any, Union

import numpy as np
from signature import Signatures
from scipy.spatial import distance
import graphviz
from dash_interactive_graphviz import DashInteractiveGraphviz


class Node():
    """ Node class for the tree """

    def __init__(self, identifier: Union[str, tuple], data: Any = None, level: int = -1):
        self.identifier: Union[str, tuple[float]] = identifier
        self.data: Any = data
        self.children: list["Node"] = []
        self.label: str = ""
        self.original_level: int = level
        self.all_levels: list[int] = [self.original_level]
        self.merge_globbed: list["Node"] = []
        self.prune_globbed: list["Node"] = []

    def add_child(self, node: "Node") -> None:
        """ Add a child to the node

        Args:
            node (Node): The node to add as a child

        Raises:
            ValueError: If the node already exists in the children
        """

        if node not in self.children:
            self.children.append(node)
        else:
            raise ValueError("Node already exists in children")

    def __repr__(self) -> str:
        return f"Node({self.label}, {self.data})"

    def __eq__(self, node: "Node") -> bool:
        return self.identifier == node.identifier

    def to_dict(self) -> dict[str, Any]:
        """ Convert the node to a dictionary

        Returns:
            dict[str, Any]: The node as a dictionary
        """
        return {
            str(self.identifier): {
                "data": self.data,
                "children": [child.to_dict() for child in self.children]
            }
        }

    def remove(self, node: "Node") -> None:
        """ Remove a child from the node

        Args:
            node (Node): The node to remove
        """
        self.children.remove(node)


class Tree():

    def __init__(
        self,
        verbose: bool = True,
        merge_threshold: float = 0.1,
        prune_threshold: float = 0.1
    ):
        self.nodes = defaultdict(Node)
        self.root = None
        self.leaves = []
        self.merge_threshold = merge_threshold
        self.prune_threshold = prune_threshold
        self.count = 0
        self.verbose = verbose

    def add_node(self, node: Node, parent: Union[Node, Union[str, tuple]] = None) -> None:
        """ Add a node to the tree

        Args:
            node (Node): The node to add
            parent (Union[Node, Union[str, tuple]], optional): The parent node. Defaults to None.

        Raises:
            ValueError: If the tree already has a root
        """

        if parent:
            if isinstance(parent, (str, tuple)):
                parent = self.get_node(parent)
            parent.add_child(node)
            node.original_level = parent.original_level + 1
            node.all_levels = [parent.original_level + 1]
        else:
            if self.root:
                raise ValueError("Tree already has a root")
            self.root = node
        self.nodes[node.identifier] = node

    def contains(self, identifier: str) -> bool:
        """_summary_

        Args:
            identifier (str): Node identifier

        Returns:
            bool: Whether the tree contains the node or not
        """
        return any(node == identifier for node in self.nodes.keys())

    def get_node(self, identifier: str) -> Node:
        """ Get a node from the tree

        Args:
            identifier (str): Node identifier

        Returns:
            Node: The node
        """
        return self.nodes[identifier]
    
    def get_node_by_label(self, label: str) -> Node:
        """ Get a node from the tree by label

        Args:
            label (str): Node label

        Returns:
            Node: The node
        """
        if not isinstance(label, str):
            label = str(label)

        for node in self.nodes.values():
            if node.label == label:
                return node
        return None

    def get_node_by_level(self, level: int) -> list[Node]:
        """ Get a node from the tree by level

        Args:
            level (int): Node level]

        Returns:
            list[Node]: The list of nodes
        """
        return [node for node in self.nodes.values() if level in node.all_levels]


    def get_all_signature_by_branch(self, branch: Any) -> list[Node]:
        """ Get a signatures from the tree by branch

        Args:
            branch (Any)

        Returns:
            list[Node]: The list of nodes
        """
        branch_unsorted = [node for node in self.nodes.values() if branch in node.data]
        branch_sorted = []
        for k in range(len(branch_unsorted)):
            for node in branch_unsorted:
                if k in node.all_levels:
                    branch_sorted.append(node.identifier)

        return branch_sorted

    def to_dict(self) -> dict[str, Any]:
        """ Convert the tree to a dictionary

        Returns:
            dict[str, Any]: The tree as a dictionary
        """
        return self.root.to_dict()

    def get_node_info(self, node: Node):
        try:
            node_name = list(node.keys())[0]
            node_children = node[node_name].get('children', None)
        except AttributeError:
            node_name = node
            node_children = None
        return node_name, node_children

    def get_tree_size_and_depth(self, node: Node, depth: int = 0):
        _, children = self.get_node_info(node)
        if children is None:
            return 1, depth
        node_count = 1
        max_depth = depth
        for child in children:
            child_count, child_depth = self.get_tree_size_and_depth(
                child,
                depth + 1
            )
            node_count += child_count
            max_depth = max(max_depth, child_depth)
        return node_count, max_depth

    def naming(self, node: Node = None, name: str = None) -> None:
        if name is None:
            siblings = []
            if node is None:
                node = self.root
                self.root.label = str(self.count)

                for sibling in node.children:
                    self.count += 1
                    sibling.label = str(self.count)
                
                for child in node.children:
                    for child_2nd in child.children:
                        siblings.append(child_2nd)

            elif isinstance(node, list):
                for sibling in node:
                    self.count += 1
                    sibling.label = str(self.count)

                for _node in node:
                    for child in _node.children:
                        siblings.append(child)

            if len(siblings) > 0:
                self.naming(siblings)
                
        elif node is not None and name is not None:
            node.label = name
        else:
            raise ValueError("Both node and name must be provided or neither")

    def merge(self, node: Node = None, reduce: str = "average") -> None:
        """ Merge siblings node that are close. """
        if reduce not in ["average", "static"]:
            raise ValueError("Reduce must be either average or static")

        if node is None:
            node = self.root
        
        for i, child in enumerate(node.children):
            for sibling in node.children[i + 1:]:
                node_distance = distance.euclidean(
                    np.array(child.identifier),
                    np.array(sibling.identifier)
                )
                if node_distance < self.merge_threshold:
                    if self.verbose:
                        message = f"Merging {child.label} and "
                        message += f"{sibling.label} distance {node_distance}"
                        print(message)
                    child.data.extend(sibling.data)
                    child.data = list(set(child.data))
                    child.children.extend(sibling.children)
                    child.merge_globbed.append(sibling)
                    node.remove(sibling)
                    self.nodes.pop(sibling.identifier, 1)
                    
                    if reduce == "average":
                        st_signature = np.array(child.identifier)
                        nd_signature = np.array(sibling.identifier)
                        new_identifier = np.mean([st_signature, nd_signature], axis=0)
                        new_identifier = tuple(new_identifier.tolist())
                        child.identifier = new_identifier
            self.merge(child, reduce)

    def prune(self, node = None) -> None:
        """ Prune children that are close. """
        changed = False

        if node is None:
            node = self.root

        for i, child in enumerate(node.children):
            node_distance = distance.euclidean(
                np.array(node.identifier),
                np.array(child.identifier)
            )
            if node_distance < self.prune_threshold:
                changed = True
                if self.verbose:
                    message = f"Prunning {child.label} into "
                    message += f"{node.label} distance {node_distance}"
                    print(message)

                node.data.extend(child.data)
                node.data = list(set(node.data))
                node.children.extend(child.children)
                node.prune_globbed.append(child)
                node.all_levels.append(child.original_level)
                node.remove(child)
                #self.nodes.remove(child)
                self.nodes.pop(child.identifier, 1)

        if changed:
            self.prune(node)
        else:
            for child in node.children:
                self.prune(child)

    def render(
        self,
        path: str = "./",
        name: str = "tree.gv",
        save: bool = True
    ) -> tuple[DashInteractiveGraphviz, str]:
        """Render the tree"""
        node_count, tree_depth = self.get_tree_size_and_depth(self.to_dict())
        width = max(10, node_count * 0.1)
        height = max(10, tree_depth * 0.5)

        if not os.path.exists(path):
            os.makedirs(path)

        if path[-1] != "/":
            path += "/"

        f = graphviz.Digraph('signature_tree', filename=f'{path}{name}', format='pdf')
        f.attr(rankdir='TB', size=f'{width},{height}', nodesep='0.5', ranksep='1')
        f.attr('node', shape='rectangle')

        traversed_nodes = [self.root]
        while traversed_nodes:
            cur_node = traversed_nodes.pop(0)
            cur_node_name = str(cur_node.label)

            f.node(cur_node_name, label=cur_node.label)

            for next_node in cur_node.children:
                traversed_nodes.append(next_node)
                next_node_name = str(next_node.label)
                f.edge(cur_node_name, next_node_name)

        if save:
            f.render()
        return DashInteractiveGraphviz(id="signature_path", dot_source=f.source), f.source


if __name__ == "__main__":
    sys.setrecursionlimit(9999)
    from datetime import datetime

    start = datetime.now()
    depth = 4
    x = Signatures(device="cpu", depth=depth)
    print(f"Time in seconds taken for depth {depth}: {(datetime.now() - start).total_seconds()}")

    start = datetime.now()
    tree = Tree(prune_threshold=1)
    root = Node(identifier=(1,), data=[], level=0)
    tree.add_node(root)

    for t, signatures in x.signatures.items():
        previous = root
        for signature in signatures:
            if not tree.contains(signature):
                node = Node(identifier=signature, data=[t])
                tree.add_node(node, None if signature == (1,) else previous)
            else:
                node = tree.get_node(signature)
                node.data.append(t)
                node.data = list(set(node.data))

            previous = node
    print(f"Time in seconds taken for tree construction: {(datetime.now() - start).total_seconds()}")
    

    tree.naming()
    tree.merge()
    # tree.prune()
    tree.render("output", "tree.gv")

    level_ord = []
    k = 0
    nodes = tree.get_node_by_level(k)
    while len(nodes) > 0:
        level_k = []
        for node in nodes:
            level_k.append(node.identifier)
        level_ord.append(level_k)
        k = k + 1
        nodes = tree.get_node_by_level(k)

    for l in level_ord:
        print(len(l))


    # tree.merge()

    # n = tree.get_node((1,))

    # print(n)

    # bla
    # tree.render("output", "tree.gv")