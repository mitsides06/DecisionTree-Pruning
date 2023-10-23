"""
Introduction to Machine Learning
CW1
"""
import numpy as np
import matplotlib.pyplot as plt
import pprint



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


def evaluate(test_data, tree):
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
    return correct / len(test_data)




# Step 4:
def is_node_connected_to_leaves(node):
    """
    Check if a node is directly connected to two leaves.
    """
    return not node["leaf"] and node["left"]["leaf"] and node["right"]["leaf"]


def performance_difference(tree, train_data, validation_data):
    """
    Helper function to compute performance difference.
    """
    return abs(evaluate(train_data, tree) - evaluate(validation_data, tree))


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
        current_performance_difference = performance_difference(root, full_train_data, validation_data)

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
        new_performance_difference = performance_difference(root, full_train_data, validation_data)
        print("pruning")

        # Revert pruning if it didn't decrease the difference:
        if new_performance_difference > current_performance_difference: 	# Big difference if > or >=  *see remarks at the end
            node.clear()
            node.update(original_node)
            print("but pruning unsuccessful")




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



if __name__ == "__main__":
	# Example usage:
	train_data = clean_data
	validation_data = noisy_data

	#train_data = data_clean[:int(len(clean_data) * 0.7)]
	#validation_data = data_clean[int(len(clean_data) * 0.7):int(len(clean_data) * 0.85)]
	#test_data = data_clean[int(len(clean_data) * 0.85):]

	# Create the decision tree
	tree, depth = decision_tree_learning(train_data)

	# Evaluate the original tree
	accuracy = evaluate(train_data, tree)
	print("Accuracy of original tree on train_data:", accuracy)
	accuracy = evaluate(validation_data, tree)
	print("Accuracy of original tree on validation_data:", accuracy)
	#pprint.pp(tree)
	print()
	plot_tree(tree)

	# Prune the tree
	node = tree
	subset_train_data = train_data.copy()
	prune_tree(tree, node, train_data, subset_train_data, validation_data)

	# Evaluate the pruned tree
	accuracy = evaluate(train_data, tree)
	print("Accuracy of pruned tree on train_data:", accuracy)
	accuracy = evaluate(validation_data, tree)
	print("Accuracy of pruned tree on validation_data:", accuracy)
	#pprint.pp(tree)
	plot_tree(tree)




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


