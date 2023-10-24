"""
Introduction to Machine Learning
CW1
"""
import numpy as np
import matplotlib.pyplot as plt
import pprint
import random



# Step 1: Loading data
clean_data = np.loadtxt("wifi_db/clean_dataset.txt")
noisy_data = np.loadtxt("wifi_db/noisy_dataset.txt")	# should process the labels - round them to integers/




# Step 2: Creating Decision Trees
def entropy(data):
    """
    Calculate the entropy of the data

    Args:
        data (list): dataset

    Returns:
        numpy.float64: entropy value 
    """

    labels = data[:, -1]
    _, counts = np.unique(labels, return_counts=True)
    probs = counts / len(labels)
    return -np.sum(probs * np.log2(probs))


def gain(s_all, s_left, s_right):
    """
    Calculate the information gain of the subsets (s_left, s_right) associated with the dataset (s_all)

    Args:
        s_all (list): dataset
        s_left (list): subset on the left
        s_right (list): subset on the right 

    Returns:
        numpy.float64: information gain: the difference between the initial entropy and the average entropy of the produced subsets
    """

    return entropy(s_all) - (len(s_left)/len(s_all)*entropy(s_left) + len(s_right)/len(s_all)*entropy(s_right))


def find_split(data):
    """
    Chooses the attribute and the value that results in the highest information gain

    Args:
        data (list): the dataset

    Returns:
        tuple: (an attribute of the dateset, optimal value to split in the attribute)
    """

    best_gain = 0
    best_split = None

    for feature in range(data.shape[1] - 1):
        unique_values = np.unique(data[:, feature])
        for value in unique_values[1:]:  # we don't need to go over the last unique value as the right split will be empty: yessir! Also changed for the node to have value: x<value not x<=value
            left_split = get_left_split_data(data, feature, value)
            right_split = get_right_split_data(data, feature, value)

            current_gain = gain(data, left_split, right_split)

            if current_gain > best_gain:
                best_gain = current_gain
                best_split = (feature, value)

    return best_split


def decision_tree_learning(data, depth=0):
    """
    Recursive Decision Tree Algorithm

    Args:
        data (list): dataset
        depth (int, optional): the depth of the node. Defaults to 0.

    Returns:
        dict: decision tree
    """

    labels = data[:, -1]
    if len(np.unique(labels)) == 1:
        return {"leaf": True, "label": data[0, -1]}, depth  # removed "depth" from the dictionary, and put it as an element in a 2-tuple as our function should return (node, depth)

    feature, value = find_split(data)
    left_data = get_left_split_data(data, feature, value)		# Also changed for the node to have value: x<value not x<=value (related to comment in find_split())
    right_data = get_right_split_data(data, feature, value)

    left_branch, left_depth = decision_tree_learning(left_data, depth+1)
    right_branch, right_depth = decision_tree_learning(right_data, depth+1)

    return {
        "leaf": False,
        "feature": feature,
        "value": value,
        "left": left_branch,
        "right": right_branch
    }, max(left_depth, right_depth)  # did the same thing as above. I am not quite sure about the styling convection here tho - arranged styling convention (my best guess)


def get_left_split_data(data, feature, value):
	return data[data[:, feature] < value]


def get_right_split_data(data, feature, value):
	return data[data[:, feature] >= value]


# Step 3: Classification & Evaluation
def classify(instance, tree):
    """
    Classification function to classify instances in the tree

    Args:
        instance (list): a testing example
        tree (dictionary): decision tree

    Returns:
        int: the label
    """

    if tree["leaf"]:
        return tree["label"]
    
    if instance[tree["feature"]] < tree["value"]:
        return classify(instance, tree["left"])
    else:
        return classify(instance, tree["right"])


def find_accuracy(test_data, tree):
    """
    Evaluation of the algorithm

    Args:
        test_data (list): a list of example testing data
        tree (dictionary): decision tree

    Returns:
        float: percentage of accuracy
    """

    correct = 0
    for instance in test_data:
        prediction = classify(instance[:-1], tree)
        if prediction == instance[-1]:
            correct += 1
    return np.array(correct / len(test_data))


def find_confusion_matrix(testing_data, training_tree):
    """
    Find out the confusion matrix

    Args:
        testing_data (_type_): test data
        training_tree (_type_): decision tree trained on training data
        
    Returns:
        2D list: confusion matrix
    """
    prediction_labels = []
    actual_labels = [instance[-1] for instance in testing_data]
    
    # Getting the predictions of each instance in testing data
    for instance in testing_data:
        prediction = classify(instance[:-1], training_tree)
        
        prediction_labels.append(prediction)
    
    # Initating confusion matrix
    confusion_matrix = [[0 for j in range(4)] for i in range(4)]
    
    # for every prediction and actual pair, it updates the confusion matrix
    for pred, act in zip(prediction_labels, actual_labels):
        confusion_matrix[int(act)-1][int(pred)-1] += 1
    
    return np.array(confusion_matrix)
    

def find_precision(confusion_matrix):
    """
    Find out the precision

    Args:
        confusion_matrix (2D list): confusion matrix of testing data

    Returns:
        tuple: tuple of precision value per class
    """
    
    precision_per_class = []
    
    # Loop precision calculation for each class/label
    for label in range(4):
        true_positives = confusion_matrix[label][label]
        
        total_predicted = 0
        for row in confusion_matrix: # Predicted values are columns, loop is necessary to sum the elements a column
            total_predicted += row[label]
            
        precision = true_positives / total_predicted
        precision_per_class.append(precision)
        
    return np.array(precision_per_class)

def find_recall(confusion_matrix):
    """
    Find out the recall

    Args:
        confusion_matrix (2D list): confusion matrix of testing data
        
    Returns:
        tuple: tuple of recall value per class
    """
    
    recall_per_class = []
    
    # Loop recall calculation for each class/label
    for label in range(4):

        true_positives = confusion_matrix[label][label] # True values are rows, can sum row to find number of truly labelled instances
        total_true = sum(confusion_matrix[label])
        
        recall = true_positives / total_true
        recall_per_class.append(recall)
        
    return np.array(recall_per_class)

def find_f1(confusion_matrix):
    """
    Find out the recall

    Args:
        confusion_matrix (2D list): confusion matrix of testing data
        
    Returns:
        tuple: tuple of f1 value per class
    """
    # Use recall and precision functions to find F1
    recall_per_class = find_recall(confusion_matrix)
    precision_per_class = find_precision(confusion_matrix)
    
    # Calculate F1
    f1_per_class = 2 * np.multiply(recall_per_class, precision_per_class) / np.add(recall_per_class, precision_per_class)
    
    return np.array(f1_per_class)

def cross_validation(data, k=10):
    """
    1. Shuffle the data
    2. Split it into k-fold
    3. Train k times with k-1 folds as training data
    4. Evaluate (Confusion Matrix, Accuracy, Precision, Recall, F-1) each time with k fold as the testing data

    Args:
        data (2D list): dataset
        k (int, optional): number of folds. Defaults to 10.

    Returns:
        tuple: (average_confusion_matrix, average_accuracy, average_precision_per_class, average_recall_per_class, F-1)
    """
    
    shuffled_data = data
    
    # Setting the random seed
    np.random.seed(0)
    
    # Randomly shuffling the dataset
    np.random.shuffle(shuffled_data)
    
    # Separating into k folds
    fold_size = len(shuffled_data) // k
    remainder = len(shuffled_data) % k
    folds = []

    start = 0
    for i in range(k):
        end = start + fold_size + (1 if i < remainder else 0)
        fold = shuffled_data[start:end]
        folds.append(fold)
        start = end
        
      
    # Storing metrics
    confusion_matrices = []
    accuracies = []
    precisions = []
    recalls = []
    f1s = []

    for i in range(k): # REPLACED LEN(FOLDS) WITH K AS K == LEN(FOLDS) BY THE ABOVE FOR LOOP CONSTRUCTION
        testing_data = folds[i]

        training_data = np.concatenate([arr for arr in folds[:i]+folds[i+1:]], axis=0)
        #training_data = np.delete(folds, i, 0)[0]       THIS CODE GIVES AN ARRAY WITH SHAPE (200, 8), THE TRAINING_DATA SHAPE SHOULD BE (1800, 0). I REPLACED IT WITH THE ONE ABOVE, IT SHOULD BE FINE NOW BUT GIVE IT A LOOK.

        training_tree, _ = decision_tree_learning(training_data) # HAVING LEARNED THIS TRICK FROM FADI, SINCE WE DON'T NEED THE DEPTH VARIABLE I REPLACED DEPTH VARIABLE WITH _ ;)
        
        confusion_matrix = find_confusion_matrix(testing_data, training_tree)
        accuracy = find_accuracy(testing_data, training_tree) # ADDED THIS CODE
        precision = find_precision(confusion_matrix)  # ADDED THIS CODE. HAVEN'T CHECKED IF THE FIND_PRECISION FUNCTION IS CORRECT THO
        recall = find_recall(confusion_matrix)  # ADDED THIS CODE. HAVEN'T CHECKED IF THE FIND_PRECISION FUNCTION IS CORRECT THO
        f1 = find_f1(confusion_matrix) 
        
        confusion_matrices.append(confusion_matrix)   
        # APPENDED THE METRICS
        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        
    # Average the Metrics across all folds
    average_confusion_matrix = np.sum(confusion_matrices, axis=0) / len(confusion_matrices)
    average_accuracy = sum(accuracies)/len(accuracies)
    average_precision_per_class = np.sum(precisions, axis=0) / len(precisions)
    average_recall_per_class = np.sum(recalls, axis=0) / len(recalls)
    average_f1 = np.sum(f1s, axis=0) / len(f1s)
    
    print(f"The average confusion matrix is:\n{average_confusion_matrix}\nThe average accuracy is: {average_accuracy}\nThe aerage precision per class is: {average_precision_per_class}\nThe average recall per class is: {average_recall_per_class}\nThe average f_1 per class is: {average_f1} ")


# Step 4:
def is_node_connected_to_leaves(node):
    """
    Check if a node is directly connected to two leaves.
    """
    return not node["leaf"] and node["left"]["leaf"] and node["right"]["leaf"]


def performance_difference(tree, clean_test_data, noisy_test_data):      # CHANGED VARIABLE NAMES FOR CLARITY
    """
    Helper function to compute performance difference.
    """
    return abs(find_accuracy(clean_test_data, tree) - find_accuracy(noisy_test_data, tree))  


def prune_tree(root, node, full_train_data, subset_train_data, validation_data):
    """
    Prune the tree based on validation error.
    """
    # Base Case: If tree is a leaf, no pruning needed.
    if node["leaf"]:
        return root
    
    # Recursive Case: Check left and right children.
    if not node["left"]["leaf"]:
        left_data = get_left_split_data(subset_train_data, node["feature"], node["value"])
        prune_tree(root, node["left"], full_train_data, left_data, validation_data)
    if not node["right"]["leaf"]:
        right_data = get_right_split_data(subset_train_data, node["feature"], node["value"])
        prune_tree(root, node["right"], full_train_data, right_data, validation_data)
    
    # Check if current node is connected to two leaves
    if is_node_connected_to_leaves(node):
        # Now that we are at a parent node of two leaf children nodes:
        # Calculate the performance difference between train and validation data before pruning:
        current_accuracy = find_accuracy(validation_data, root) # REPLACED PERFORMANCE_DIFFERENCE WITH EVALUATE FUNCTION

        # Store the entire node to revert if needed:
        original_node = node.copy()

        # Pruning: replace the parent node by a leaf node with the majority label from the current training data subset.
        # Calculate the majority label:
        labels = subset_train_data[:, -1]
        labels, counts = np.unique(labels, return_counts=True)
        majority_label = labels[np.argmax(counts)]

        # Convert the current node to a leaf (modifications propagate to the root since both reference the same object).
        node.clear()
        node.update({
            "leaf": True,
            "label": majority_label
        })

        # Calculate the new performance difference between train and validation data after pruning:
        new_accuracy = find_accuracy(validation_data, root) # REPLACED PERFORMANCE_DIFFERENCE FUNCTION WITH EVALUATE FUNCTION
        #print("pruning")

        # Revert pruning if it didn't decrease the difference:
        if new_accuracy < current_accuracy: 	# Big difference if > or >=  *see remarks at the end
            node.clear()
            node.update(original_node)
            #print("but pruning unsuccessful")




# Bonus Part: Tree Visualisation
# Define a list of colors.
COLORS = ["blue", "green", "red", "cyan", "magenta", "orange", "black", "purple", "brown", "gray", "olive"]

def plot_tree(tree, y=0, depth=0, ax=None, x_coord_dict=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(20, 10))
        ax.axis('off')
        x_coord_dict = {"next_x": 0}
    
    color = COLORS[depth % len(COLORS)]

    # If leaf, plot it and return
    if tree["leaf"]:
        x = x_coord_dict["next_x"]
        x_coord_dict["next_x"] += 1
        ax.text(x, y, f"Leaf: {int(tree['label'])}", bbox=dict(boxstyle="round,pad=0.3", edgecolor=color, facecolor="aliceblue"), ha='center', fontsize=4)
        return x

    # Post-order traversal: First process children
    left_x = plot_tree(tree["left"], y-1, depth+1, ax, x_coord_dict)
    right_x = plot_tree(tree["right"], y-1, depth+1, ax, x_coord_dict)

    # Compute parent's x as average of children's x
    x = (left_x + right_x) / 2.0

    ax.text(x, y, f"x{tree['feature']} < {tree['value']}", bbox=dict(boxstyle="round,pad=0.3", edgecolor=color, facecolor="aliceblue"), ha='center', fontsize=4)
    
    ax.plot([x, left_x], [y-0.1, y-1+0.1], color)
    ax.plot([x, right_x], [y-0.1, y-1+0.1], color)
    
    if depth == 0:
        plt.show()
    
    return x

# INCOMPLETE
def cross_validation_after_pruning(data, k=10):    #WE NEED TO CHANGE THE PRUNE_TREE FUNCTION SO THAT IT TRACKS THE DEPTH. IT WILL BE NEEDED FOR DEPTH ANALYSIS (BEFORE VS AFTER PRUNING)
    shuffled_data = data
    
    # Setting the random seed
    np.random.seed(0)
    
    # Randomly shuffling the dataset
    np.random.shuffle(shuffled_data)
    
    # Separating into k folds
    fold_size = len(shuffled_data) // k
    remainder = len(shuffled_data) % k
    folds = []

    start = 0
    for i in range(k):
        end = start + fold_size + (1 if i < remainder else 0)
        fold = shuffled_data[start:end]
        folds.append(fold)
        start = end
        
      
    # Storing metrics
    confusion_matrices = []    
    accuracies = []
    precisions = []
    recalls = []
    f_1s = []
    models = []
    for test_idx in range(k): 
        test_data = folds[i]
       
        for valid_idx in range(k-1):
            new_folds = folds[ :test_idx] + folds[test_idx+1: ]
            valid_data = new_folds[valid_idx]
            train_data = np.concatenate([arr for arr in new_folds[:valid_idx]+new_folds[valid_idx+1:]], axis=0)
            tree, depth = decision_tree_learning(train_data)
            node = tree
            sub_train_data = train_data.copy() # FADI IS THIS NEEDED? YOU DID THAT STEP WHEN YOU CALLED THE PRUNE_TREE FUNCTION, BUT IS IT NECESSARY? (same above)
            prune_tree(tree, node, train_data, sub_train_data, valid_data)
            pruned_tree = tree

            final_accuracy = find_accuracy(test_data, pruned_tree)
            confusion_matrix = find_confusion_matrix(test_data, pruned_tree)
            precision = find_precision(confusion_matrix)
            recall = find_recall(confusion_matrix)
            f_1 = find_f1(confusion_matrix)
            
            confusion_matrices.append(confusion_matrix)
            accuracies.append(final_accuracy)
            precisions.append(precision)
            recalls.append(recall)
            f_1s.append(f_1)
            models.append(pruned_tree)
    
    
    
    average_confusion_matrix = np.sum(confusion_matrices, axis=0) / len(confusion_matrices)
    average_accuracy = sum(accuracies)/len(accuracies)
    average_precision_per_class = np.sum(precisions, axis=0) / len(precisions)
    average_recall_per_class = np.sum(recalls, axis=0) / len(recalls)
    average_f1 = np.sum(f_1s, axis=0) / len(f_1s)

    print(f"The average confusion matrix is:\n{average_confusion_matrix}\nThe average accuracy is: {average_accuracy}\nThe aerage precision per class is: {average_precision_per_class}\nThe average recall per class is: {average_recall_per_class}\nThe average f_1 per class is: {average_f1} ")

if __name__ == "__main__":
    #a, b, c, d, e = cross_validation(clean_data)
    #print(f"Confusion Matrices: {a}")
    #print(f"Accuracy: {b}")
    #print(f"Precision: {c}")
    #print(f"Recall: {d}")
    #print(f"F-1: {e}")
    print("PRE-PRUNING EVALUATION METRICS ON CLEAN DATA:\n")
    cross_validation(clean_data)
    print()
    print("PRE-PRUNING EVALUATION METRICS ON NOISY DATA:\n")
    cross_validation(noisy_data)
    print()
    print("POST-PRUNING EVALUATION METRICS ON CLEAN DATA:\n")
    cross_validation_after_pruning(clean_data)
    print()
    print("POST-PRUNING EVALUATION METRICS ON NOISY DATA:\n ")
    cross_validation_after_pruning(noisy_data)

    
#if __name__ == "__main__":
#     # Example usage:
#     train_data = clean_data
#     validation_data = noisy_data
    
     #train_data = data_clean[:int(len(clean_data) * 0.7)]
     #validation_data = data_clean[int(len(clean_data) * 0.7):int(len(clean_data) * 0.85)]
     #test_data = data_clean[int(len(clean_data) * 0.85):]
    
     # Create the decision tree
    
#     tree, depth = decision_tree_learning(train_data)
    
     # Evaluate the original tree
#     accuracy = find_accuracy(train_data, tree)
#     print("Accuracy of original tree on train_data:", accuracy)
#     accuracy = find_accuracy(validation_data, tree)
#     print("Accuracy of original tree on validation_data:", accuracy)
#     #pprint.pp(tree)
#     print()
#     plot_tree(tree)
    
     # Prune the tree
#     node = tree
#     subset_train_data = train_data.copy()
#     prune_tree(tree, node, train_data, subset_train_data, validation_data)
    
     # Evaluate the pruned tree
#     accuracy = find_accuracy(train_data, tree)
#     print("Accuracy of pruned tree on train_data:", accuracy)
#     accuracy = find_accuracy(validation_data, tree)
#     print("Accuracy of pruned tree on validation_data:", accuracy)
#     #pprint.pp(tree)
#     plot_tree(tree)



"""
*Remarks: if >, we get pruning but overall accuracy is reduced but not their difference with accuracies of 0.997[training_data] and 0.915[validation_data]
		  if >=, tree says the same with accuracy of 1.0[training_data] and 0.918[validation_data]. Note that in both cases accuracy difference is the same (=0.082)

quoting:
'The behavior you observed is consistent with the nature of pruning. Pruning is designed to reduce overfitting. By pruning the tree, you're making the model less complex, and therefore, it may not fit the training data as closely as before. This can lead to a decrease in training accuracy. The validation accuracy might also decrease, but the hope is that it doesn't decrease as much as the training accuracy, thereby reducing the gap (difference) between the two.

If you pruned with > and observed a decrease in both training and validation accuracy, but the difference between the two remains the same, it means the pruning made the model simpler without necessarily making the model generalize better to the validation set.

A few things to consider:

1) Pruning Objective: If your objective was strictly to reduce the difference between training and validation accuracy (even at the expense of a slightly lower accuracy), then the pruning did its job. However, if you were hoping for an increase in validation accuracy, you might need to adjust your pruning criteria or consider other approaches.

2) Over-Pruning: It's possible to over-prune, which would remove too many nodes and make the tree too simple, losing important decision boundaries. You can experiment with less aggressive pruning criteria.

3) Other Techniques: If pruning doesn't yield the desired results, consider other regularization techniques or even ensemble methods like random forests or gradient boosting, which are inherently less prone to overfitting.

Remember, the ultimate goal of machine learning models is good generalization performance, not perfect training performance. The reduced gap indicates better generalization, even if the absolute accuracies are slightly lower.'
"""

