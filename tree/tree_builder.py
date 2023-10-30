"""
Creating Decision Trees
"""

import numpy as np


def entropy(data):
    """ Calculate the entropy of the data.

    Args:
        data (np.array): dataset

    Returns:
        numpy.float64: entropy value
    """
    # Return 0 if the dataset is empty to avoid errors in log calculations.
    if len(data) == 0:
        return 0

    labels = data[:, -1]
    _, counts = np.unique(labels, return_counts=True)
    probs = counts / len(labels)
    return -np.sum(probs * np.log2(probs))


def gain(s_all, s_left, s_right):
    """ Calculate the information gain achieved by splitting the dataset.

    The dataset (s_all) is split into two subsets (s_left, s_right).

    Args:
        s_all (np.array): dataset
        s_left (np.array): subset on the left
        s_right (np.array): subset on the right

    Returns:
        numpy.float64: information gain: the difference between the initial
                       entropy of s_all and the weighted average entropy of
                       the produced subsets (s_left and s_right)
    """
    remainder = (len(s_left)/len(s_all)*entropy(s_left)
                 + len(s_right)/len(s_all)*entropy(s_right))
    return entropy(s_all) - remainder


def find_split(data):
    """ Chooses the attribute and the value that result in the highest
    information gain.

    Args:
        data (np.array): dataset

    Returns:
        tuple: best feature and value to split the dataset
    """
    best_gain = 0
    best_split = None

    for feature in range(data.shape[1] - 1):
        unique_values = np.unique(data[:, feature])
        for value in unique_values[1:]:
            left_split = get_left_split_data(data, feature, value)
            right_split = get_right_split_data(data, feature, value)

            current_gain = gain(data, left_split, right_split)

            if current_gain > best_gain:
                best_gain = current_gain
                best_split = (feature, value)

    return best_split


def get_left_split_data(data, feature, value):
    """ Retrieve the subset of data given by the left split.

    This is the subset where the specified feature has values strictly
    less than the provided value.

    Args:
        data (numpy.array): The dataset to split.
        feature (int): The index of the feature based on which the split
                       is determined.
        value (float): The threshold value to decide the split.

    Returns:
        numpy.array: The subset of data where the specified feature's value
                     is less than the given value.
    """
    return data[data[:, feature] < value]


def get_right_split_data(data, feature, value):
    """ Retrieve the subset of data given by the right split.

    This is the subset where the specified feature has values greater than
    or equal to the provided value.

    Args:
        data (numpy.array): The dataset to split.
        feature (int): The index of the feature based on which the split
                       is determined.
        value (float): The threshold value to decide the split.

    Returns:
        numpy.array: The subset of data where the specified feature's value
                     is greater than or equal to the given value.
    """
    return data[data[:, feature] >= value]


def decision_tree_learning(data, depth=0):
    """ Recursive Decision Tree Algorithm.

    Args:
        data (np.array): dataset
        depth (int): the depth of the node. Defaults to 0

    Returns:
        dict: A recursive dictionary representing the decision tree.
              Each node in the tree is either a leaf node or an internal node:
                 - If it's a leaf node, the dictionary has the format:
                   {
                       "leaf": True,
                       "label": label_value
                   }
                 - If it's an internal node, the dictionary has the format:
                   {
                       "leaf": False,
                       "feature": feature_index,
                       "value": split_value,
                       "left": left_subtree_dictionary,
                       "right": right_subtree_dictionary
                   }
    """
    labels = data[:, -1]
    if len(np.unique(labels)) == 1:
        return {"leaf": True, "label": data[0, -1]}, depth

    feature, value = find_split(data)
    left_data = get_left_split_data(data, feature, value)
    right_data = get_right_split_data(data, feature, value)

    left_branch, left_depth = decision_tree_learning(left_data, depth+1)
    right_branch, right_depth = decision_tree_learning(right_data, depth+1)

    return {
        "leaf": False,
        "feature": feature,
        "value": value,
        "left": left_branch,
        "right": right_branch
    }, max(left_depth, right_depth)


def find_depth(node):
    """ Compute the depth of the tree rooted at the given node.

    Args:
        node (dict): The node to start computing depth from.
                     See `decision_tree_learning` for structure details.

    Returns:
        int: The depth of the tree.
    """
    # Base case: if it's a leaf, depth is 0.
    if node['leaf']:
        return 0

    # Recursive case: compute the depths of the left and right subtrees.
    left_depth = find_depth(node['left'])
    right_depth = find_depth(node['right'])

    # The depth of the current tree is 1 (for the current node) 
    # plus the maximum depth of its children.
    return 1 + max(left_depth, right_depth)