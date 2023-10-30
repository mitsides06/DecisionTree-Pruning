"""
Pruning
"""

import numpy as np
from tree.tree_builder import get_left_split_data, get_right_split_data 
from evaluation.classification_evaluation import find_accuracy 


def is_node_connected_to_leaves(node):
    """ Check if a node is directly connected to two leaves.

    Args:
        node (dict): The node to check.

    Returns:
        bool: True if the node is directly connected to two leaves,
              False otherwise.
    """
    return not node["leaf"] and node["left"]["leaf"] and node["right"]["leaf"]


def prune_tree(root, node, node_data_subset, validation_data):
    """ Prune the tree based on validation accuracy.

    Args:
        root (dict): The root of the decision tree.
                     See `decision_tree_learning` in tree.tree_builder
                     for structure details.
        node (dict): The current node to consider for pruning.
        node_data_subset (np.array): Training data subset for the current node.
        validation_data (np.array): Validation data for the tree.
    """
    # Base Case: If tree is a leaf, no pruning needed.
    if node["leaf"]:
        return

    # Recursive Case: Check left and right children.
    if not node["left"]["leaf"]:
        left_data = get_left_split_data(node_data_subset,
                                        node["feature"],
                                        node["value"])
        prune_tree(root, node["left"], left_data, validation_data)
    if not node["right"]["leaf"]:
        right_data = get_right_split_data(node_data_subset,
                                          node["feature"],
                                          node["value"])
        prune_tree(root, node["right"], right_data, validation_data)

    # Check if current node is connected to two leaves
    if is_node_connected_to_leaves(node):
        # Now that we are at a parent node of two leaf children nodes:
        # Calculate the accuracy before pruning:
        current_accuracy = find_accuracy(validation_data, root)

        # Store the entire node to revert if needed:
        original_node = node.copy()

        # Pruning: replace the parent node by a leaf node with the
        # majority label from the current training data subset.
        # > Calculate the majority label:
        labels = node_data_subset[:, -1]
        labels, counts = np.unique(labels, return_counts=True)
        majority_label = labels[np.argmax(counts)]

        # > Convert the current node to a leaf (modifications propagate to
        # the root since both reference the same object).
        node.clear()
        node.update({
            "leaf": True,
            "label": majority_label
        })

        # Calculate the new accuracy after pruning:
        new_accuracy = find_accuracy(validation_data, root)

        # Revert pruning if it didn't increase or result in the same accuracy:
        if new_accuracy < current_accuracy:
            node.clear()
            node.update(original_node)

