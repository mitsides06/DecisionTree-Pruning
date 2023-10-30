"""
Classification & Evaluation
"""

import numpy as np


def classify(instance, tree):
    """ Classify an instance using the given decision tree.

    This function recursively traverses the decision tree based on the
    instance's feature values until it reaches a leaf node.

    Args:
        instance (np.array): A testing example
        tree (dict): The decision tree (a recursive dictionary)
                     See `decision_tree_learning` in tree.tree_builder
                     for structure details.

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
                     See `decision_tree_learning` in tree.tree_builder
                     for structure details.

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
                     See `decision_tree_learning` in tree.tree_builder
                     for structure details.
        
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


def evaluate(test_data, trained_tree):
    """ Evaluate the trained tree's performance on the test data.

    Compute the evaluation metrics based on the trained tree and the test
    data. These metrics are: the confusion matrix, the accuracy, the precision
    per class, the recall per class, and the f1 measure per class.

    Args:
        test_data (np.array): Test data array.
        trained_tree: A trained decision tree.
                      See `decision_tree_learning` in tree.tree_builder
                     for structure details.

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