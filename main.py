"""
Introduction to Machine Learning
CW1 - Decision Tree

Team members:
> Kyoya Higashino (kh123)
> Jack Hau (jhh23)
> Fadi Zahar (fz221)
> Konstantinos Mitsides (km2120)
"""

import numpy as np
from tree.tree_builder import decision_tree_learning
from evaluation.cross_validation import (cross_validation_before_pruning,
                                         cross_validation_after_pruning)
from visualisation.visualiser import plot_tree



def load_data(filepath):
    """ Load dataset from the specified path.

    Args:
        filepath (str): Path to the dataset.

    Returns:
        np.array: The loaded dataset.
    """
    try:
        data = np.loadtxt(filepath)
        return data
    except FileNotFoundError:
        print(f"Filepath {filepath} not found.")
        exit(1)
    except OSError as e:
        print(f"Unexpected error occurred while loading data from "
              f"{filepath}: {e}")
        exit(1)


if __name__ == "__main__":
    # Load the clean and noisy datasets:
    clean_data_filepath = "wifi_db/clean_dataset.txt"
    noisy_data_filepath = "wifi_db/noisy_dataset.txt"
    clean_data = load_data(clean_data_filepath)
    noisy_data = load_data(noisy_data_filepath)


    # Pre-pruning evaluation on clean data:
    print("\n\nPRE-PRUNING EVALUATION METRICS ON CLEAN DATA:\n")
    cross_validation_before_pruning(clean_data)

    # Pre-pruning evaluation on noisy data:
    print("\n\nPRE-PRUNING EVALUATION METRICS ON NOISY DATA:\n")
    cross_validation_before_pruning(noisy_data)

    # Post-pruning evaluation on clean data:
    print("\n\nPOST-PRUNING EVALUATION METRICS ON CLEAN DATA:\n")
    cross_validation_after_pruning(clean_data)

    # Post-pruning evaluation on noisy data:
    print("\n\nPOST-PRUNING EVALUATION METRICS ON NOISY DATA:\n ")
    cross_validation_after_pruning(noisy_data)


    # Training and plotting decision tree on the entire clean dataset:
    train_data = clean_data
    tree, depth = decision_tree_learning(train_data)
    print("\n\nThe tree trained on the entire clean dataset has been plotted. "
          f"It has a depth of {depth}.")
    plot_tree(tree)
