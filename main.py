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
import matplotlib.pyplot as plt




# Step 1: Loading data
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
        print(f"Unexpected error occurred while loading data from {filepath}: {e}")
        exit(1)




# Step 2: Creating Decision Trees
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




# Step 3: Classification & Evaluation
def classify(instance, tree):
    """ Classify an instance using the given decision tree.

    This function recursively traverses the decision tree based on the
    instance's feature values until it reaches a leaf node.

    Args:
        instance (np.array): A testing example
        tree (dict): The decision tree (a recursive dictionary)
                     See `decision_tree_learning` for structure details.

    Returns:
        int: The label/class assigned to the instance by the tree.
    """
    # If the current tree node is a leaf, return the stored label.
    if tree["leaf"]:
        return tree["label"]

    # Depending on instance's feature value, traverse left or right subtree.
    if instance[tree["feature"]] < tree["value"]:
        return classify(instance, tree["left"])
    else:
        return classify(instance, tree["right"])


def find_accuracy(test_data, tree):
    """ Evaluate the accuracy of the decision tree on the provided test data.

    Args:
        test_data (np.array): Array of testing examples. Each instance is
                              expected to be an array where the last element
                              is the label.
        tree (dict): Trained decision tree (a recursive dictionary)
                     See `decision_tree_learning` for structure details.

    Returns:
        float: The accuracy of the decision tree on the test data as a
               percentage.
    """
    correct = 0
    for instance in test_data:
        prediction = classify(instance[:-1], tree)
        if prediction == instance[-1]:
            correct += 1
    return correct / len(test_data)


def find_confusion_matrix(test_data, tree):
    """ Compute the confusion matrix based on predictions from the tree.

    In the resulting matrix:
    - Rows represent the actual classes.
    - Columns represent the predicted classes.

    Args:
        test_data (np.array): Array of testing examples. Each instance is
                              expected to be an array where the last element
                              is the label.
        tree (dict): Trained decision tree (a recursive dictionary)
                     See `decision_tree_learning` for structure details.
        
    Returns:
        np.array: confusion matrix
    """
    actual_labels = [instance[-1] for instance in test_data]
    predicted_labels = [classify(instance[:-1], tree) for instance in test_data]

    # Get the number of unique labels (classes) from the test data.
    num_labels = len(set(actual_labels))
    
    # Initiating confusion matrix.
    confusion_matrix = np.zeros((num_labels, num_labels), dtype=int)
    
    # For every predicted and actual labels pair, update the confusion matrix.
    for pred, act in zip(predicted_labels, actual_labels):
        confusion_matrix[int(act)-1][int(pred)-1] += 1
    
    return confusion_matrix
    

def find_precision(confusion_matrix):
    """ Calculate the precision for each class based on the confusion matrix.

    Args:
        confusion_matrix (np.array): A confusion matrix of testing data.

    Returns:
        np.array: An array of precision values, one for each class.
    """
    # Diagonal values of the confusion matrix are the correct predictions.
    true_positive_counts = np.diag(confusion_matrix)
    # Summing columns for the total number of predictions made for each class.
    total_predictions_per_class = np.sum(confusion_matrix, axis=0)

    # Ensuring no cases of division by zero.
    total_predictions_per_class[total_predictions_per_class == 0] = 1

    precision_values = true_positive_counts / total_predictions_per_class

    return precision_values


def find_recall(confusion_matrix):
    """ Calculate the recall for each class based on the confusion matrix.

    Args:
        confusion_matrix (np.array): A confusion matrix of testing data.
        
    Returns:
        np.array: An array of recall values, one for each class.
    """
    # Diagonal values of the confusion matrix are the correct predictions.
    true_positive_counts = np.diag(confusion_matrix)
    # Summing rows for the total actual instances of each class.
    total_actual_per_class = np.sum(confusion_matrix, axis=1)

    # Ensuring no cases of division by zero.
    total_actual_per_class[total_actual_per_class == 0] = 1

    recall_values = true_positive_counts / total_actual_per_class

    return recall_values


def find_f1(confusion_matrix):
    """ Calculate the F1-score for each class based on the confusion matrix.

    Args:
        confusion_matrix (np.array): A confusion matrix of testing data.
        
    Returns:
        np.array: An array of F1-score values, one for each class.
    """
    # Get recall and precision values using their respective functions.
    recall_values = find_recall(confusion_matrix)
    precision_values = find_precision(confusion_matrix)

    # Ensure that the denominator isn't zero.
    denominator = np.add(recall_values, precision_values)
    denominator[denominator == 0] = 1  # avoid division by zero

    # Calculate F1-score for each class.
    f1_values = 2 * np.multiply(recall_values, precision_values) / denominator

    return f1_values


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


def evaluate(test_data, trained_tree):
    """ Evaluate the trained tree's performance on the test data.

    Compute the evaluation metrics based on the trained tree and the test
    data. These metrics are: the confusion matrix, the accuracy, the precision
    per class, the recall per class, and the f1 measure per class.

    Args:
        test_data (np.array): Test data array.
        trained_tree: A trained decision tree.
                      See `decision_tree_learning` for structure details.

    Returns:
        tuple:
            - np.array: confusion_matrix
            - float: accuracy
            - np.array: precision (per class)
            - np.array: recall (per class)
            - np.array: f1 (per class)
    """
    confusion_matrix = find_confusion_matrix(test_data, trained_tree)
    accuracy = find_accuracy(test_data, trained_tree)
    precision = find_precision(confusion_matrix)
    recall = find_recall(confusion_matrix)
    f1 = find_f1(confusion_matrix)
    return confusion_matrix, accuracy, precision, recall, f1


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




# Step 4: Pruning (and evaluation again)
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
                     See `decision_tree_learning` for structure details.
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




# Bonus Part: Tree Visualisation
def plot_tree(tree, y=0, depth=0, axis=None, x_coord_dict=None,
              save_filename='./tree_plot.png'):
    COLORS = ["blue", "green", "red", "cyan", "magenta", "orange", "black",
              "purple", "brown", "gray", "olive"]

    if axis is None:
        fig, axis = plt.subplots(figsize=(22, 10))
        axis.axis('off')
        x_coord_dict = {"next_x": 0}

    color = COLORS[depth % len(COLORS)]

    # If leaf, plot it and return
    if tree["leaf"]:
        x = x_coord_dict["next_x"]
        x_coord_dict["next_x"] += 1
        axis.text(
            x, y, f"Leaf: {int(tree['label'])}",
            bbox=dict(boxstyle="round,pad=0.3", edgecolor=color,
                      facecolor="aliceblue"),
            ha='center', fontsize=6
        )
        return x

    # Post-order traversal: First process children
    left_x = plot_tree(tree["left"], y - 1, depth + 1, axis, x_coord_dict)
    right_x = plot_tree(tree["right"], y - 1, depth + 1, axis, x_coord_dict)

    # Compute parent's x as average of children's x
    x = (left_x + right_x) / 2.0
    axis.text(
        x, y, f"x{tree['feature']} < {tree['value']}",
        bbox=dict(boxstyle="round,pad=0.3", edgecolor=color,
                  facecolor="aliceblue"),
        ha='center', fontsize=6
    )
    axis.plot([x, left_x], [y - 0.1, y - 1 + 0.1], color)
    axis.plot([x, right_x], [y - 0.1, y - 1 + 0.1], color)

    if depth == 0:
        plt.savefig(save_filename, dpi=300)  # Save with high resolution
        plt.show()

    return x




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
