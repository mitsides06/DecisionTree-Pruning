"""
Cross-Validation
"""

import numpy as np
from tree.tree_builder import decision_tree_learning, find_depth
from tree.pruning import prune_tree
from evaluation.classification_evaluation import evaluate


def cross_validation_before_pruning(data, k=10):
    """ Perform k-fold cross-validation on the data.

    The model is evaluated using various metrics:
    - Confusion Matrix
    - Accuracy
    - Precision rate per class
    - Recall rate per class
    - F-1 measure per class

    The process:
    1. Shuffle the data.
    2. Split it into k-fold.
    3. For each fold:
       - Train using the other k-1 folds.
       - Evaluate using the current fold.

    Args:
        data (np.array): The dataset to be used.
        k (int): The number of desired folds. Defaults to 10.

    Returns:
        tuple: Metrics averaged across all folds including:
               - np.array: average_confusion_matrix
               - float: average_accuracy
               - np.array: average_precision_per_class
               - np.array: average_recall_per_class
               - np.array: average_f1_per_class
    """
    # Shuffle the data
    shuffled_data = shuffle_data(data)

    # Split into k-folds
    folds = split_into_folds(shuffled_data, k)

    # Lists to store evaluation metrics for each fold
    confusion_matrices = []
    accuracies = []
    precisions = []
    recalls = []
    f1s = []

    # Compute evaluation metrics for each fold and update corresponding lists.
    for test_fold_idx in range(k):
        test_data, train_data = get_datasets_from_fold(folds, test_fold_idx)
        # Train tree on train_data
        trained_tree, _ = decision_tree_learning(train_data)

        # Compute metrics.
        (confusion_matrix,
         accuracy,
         precision,
         recall,
         f1) = evaluate(test_data, trained_tree)

        # Update corresponding lists.
        confusion_matrices.append(confusion_matrix)
        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    # Average the metrics across the k folds
    averaged_metrics = get_metrics_average(confusion_matrices,
                                           accuracies,
                                           precisions,
                                           recalls,
                                           f1s)

    print_metrics(averaged_metrics)

    return averaged_metrics


def shuffle_data(data):
    """ Shuffle the given dataset in-place.

    Args:
        data (np.array): The dataset to be shuffled.

    Returns:
        np.array: The shuffled dataset.
    """
    shuffled_data = data.copy()
    # Setting the seed for reproducibility.
    np.random.seed(0)
    # Randomly shuffle of the dataset.
    np.random.shuffle(shuffled_data)
    return shuffled_data


def split_into_folds(data, k):
    """ Split the given dataset into k consecutive folds.

    If the dataset size is not perfectly divisible by k, the initial
    folds will have one additional element.

    Args:
        data (np.array): The dataset to be split.
        k (int): The number of desired folds.

    Returns:
        list of np.array: List of data folds.
    """
    fold_size = len(data) // k
    remainder = len(data) % k

    folds = []
    start = 0
    for i in range(k):
        end = start + fold_size + (1 if i < remainder else 0)
        fold = data[start:end]
        folds.append(fold)
        start = end

    return folds


def get_datasets_from_fold(folds, test_fold_idx):
    """ Extract test and training datasets from the fold in question.

    Given the list of data folds and an index indicating the test fold, this
    function separates the test dataset from the remaining training dataset.

    Args:
        folds (list of np.array): List of data arrays, each representing a fold.
        test_fold_idx (int): Index of the fold to be used for testing.

    Returns:
        tuple:
            - np.array: test_data - Data for testing.
            - np.array: train_data - Concatenated data from all other folds
                                     for training.
    """
    test_data = folds[test_fold_idx]
    train_data = np.concatenate(
        folds[:test_fold_idx] + folds[test_fold_idx + 1:], axis=0
    )
    return test_data, train_data


def get_metrics_average(confusion_matrices,
                        accuracies,
                        precisions,
                        recalls,
                        f1s):
    """ Compute average metrics across the k-folds.

    Args:
        confusion_matrices (list of np.array): Confusion matrices list.
        accuracies (list of float): Accuracies list.
        precisions (list of np.array): Precisions list.
        recalls (list of np.array): Recalls list.
        f1s (list of np.array): F1 scores list.

    Returns:
        tuple: Averaged metrics:
            - np.array: average_confusion_matrix
            - float: average_accuracy
            - np.array: average_precision_per_class
            - np.array: average_recall_per_class
            - np.array: average_f1_per_class
    """
    average_confusion_matrix = np.mean(confusion_matrices, axis=0)
    average_accuracy = np.mean(accuracies)
    average_precision_per_class = np.mean(precisions, axis=0)
    average_recall_per_class = np.mean(recalls, axis=0)
    average_f1_per_class = np.mean(f1s, axis=0)

    return (
        average_confusion_matrix,
        average_accuracy,
        average_precision_per_class,
        average_recall_per_class,
        average_f1_per_class
    )


def print_metrics(metrics):
    """ Display the computed metrics in a formatted manner.

    Args:
        metrics (tuple): Metrics to display (confusion_matrix, accuracy,
                        precision, recall, f1).

    Returns:
        None
    """
    confusion_matrix, accuracy, precision, recall, f1 = metrics
    print(f"The average confusion matrix is:\n{confusion_matrix}\n"
          f"The average accuracy is: {accuracy}\n"
          f"The average precision per class is: {precision}\n"
          f"The average recall per class is: {recall}\n"
          f"The average f1 per class is: {f1}")


def cross_validation_after_pruning(data, k=10):
    """ Perform a nested k-fold cross-validation with pruning on the data.

    The decision tree model undergoes pruning via an additional layer of
    cross-validation. This leads to 90 pruned trees across the k-folds
    (k x (k-1) = 10 x 9) as for each outer fold, 9 inner folds are used to
    prune the tree. The model is then assessed using:

    - Confusion Matrix
    - Accuracy
    - Precision rate per class
    - Recall rate per class
    - F-1 measure per class

    Process:
    1. Shuffle the data.
    2. Split it into k-fold.
    3. For each fold:
       - Split remaining data into k-1 inner folds.
       - For each inner fold:
          a. Train with the other k-2 folds.
          b. Validate on current fold and prune tree.
          - Evaluate pruned tree using the outer test fold.

    Args:
        data (np.array): Dataset to be used.
        k (int): Desired number of folds. Defaults to 10.

    Returns:
        tuple:
               - np.array: average_confusion_matrix
               - float: average_accuracy
               - np.array: average_precision_per_class
               - np.array: average_recall_per_class
               - np.array: average_f1_per_class
        tuple:
               - float: Average pre-pruning tree depth
               - float: Average post-pruning tree depth
    """
    # Shuffle the data
    shuffled_data = shuffle_data(data)

    # Split into k-folds
    folds = split_into_folds(shuffled_data, k)

    # Lists to store evaluation metrics for each fold
    confusion_matrices = []
    accuracies = []
    precisions = []
    recalls = []
    f1s = []
    pre_pruning_depths = []
    post_pruning_depths = []

    # Compute evaluation metrics for each fold and update corresponding lists.
    for test_fold_idx in range(k):
        test_data, inner_data = get_datasets_from_fold(folds,
                                                        test_fold_idx)
        inner_folds = split_into_folds(inner_data, k - 1)
        for valid_fold_idx in range(k - 1):
            valid_data, train_data = get_datasets_from_fold(inner_folds,
                                                            valid_fold_idx)

            # Train tree on train_data and record its depth (before pruning)
            tree, pre_pruning_depth = decision_tree_learning(train_data)
            pre_pruning_depths.append(pre_pruning_depth)

            # Prune the tree using training and validation data
            prune_tree(tree, tree, train_data, valid_data)

            # Store post-pruning depth
            post_pruning_depth = find_depth(tree)
            post_pruning_depths.append(post_pruning_depth)

            # Compute metrics for pruned tree
            (confusion_matrix,
             accuracy,
             precision,
             recall,
             f1) = evaluate(test_data, tree)

            confusion_matrices.append(confusion_matrix)
            accuracies.append(accuracy)
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)

    # Average the metrics across the (k x (k-1)) folds
    averaged_metrics = get_metrics_average(confusion_matrices,
                                           accuracies,
                                           precisions,
                                           recalls,
                                           f1s)
    averaged_depths = (np.mean(pre_pruning_depths),
                       np.mean(post_pruning_depths))

    # Print averaged metrics along with average tree depths
    print(f"The average pre-pruning tree depth is: {averaged_depths[0]}")
    print(f"The average post-pruning tree depth is: {averaged_depths[1]}")
    print_metrics(averaged_metrics)

    return averaged_metrics, averaged_depths