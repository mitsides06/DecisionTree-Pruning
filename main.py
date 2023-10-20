"""
Introduction to Machine Learning
CW1 test 
"""
import numpy as np
import matplotlib.pyplot as plt



# Step 1: Loading data
try:
    data_clean = np.loadtxt("wifi_db/clean_dataset.txt")
    data_noisy = np.loadtxt("wifi_db/noisy_dataset.txt")
except FileNotFoundError:
    print("Error: Pathnames do not point to existing files")
except ValueError:
    print("Error: File contents invalid. Check for common errors such as inconsistent delimiters, number of cols, or missing elements.")




# Step 2: Creating Decision Trees
def entropy(data):
    labels = data[:, -1]
    _, counts = np.unique(labels, return_counts=True)
    probs = counts / len(labels)
    return -np.sum(probs * np.log2(probs))


def gain(s_all, s_left, s_right):
    return entropy(s_all) - (len(s_left)/len(s_all)*entropy(s_left) + len(s_right)/len(s_all)*entropy(s_right))		# maybe replace len(s) by s.shape[0] for clarity


def find_split(data):
    best_gain = 0
    best_split = None

    for feature in range(data.shape[1] - 1):
        unique_values = np.unique(data[:, feature])
        for value in unique_values[1:]:  # we don't need to go over the last unique value as the right split will be empty: yessir! Also changed for the node to have value: x<value not x<=value
            left_split = data[data[:, feature] < value]
            right_split = data[data[:, feature] >= value]

            current_gain = gain(data, left_split, right_split)

            if current_gain > best_gain:
                best_gain = current_gain
                best_split = (feature, value)

    return best_split


def decision_tree_learning(data, depth=0):
    labels = data[:, -1]
    if len(np.unique(labels)) == 1:
        return {"leaf": True, "label": data[0, -1]}, depth  # removed "depth" from the dictionary, and put it as an element in a 2-tuple as our function should return (node, depth)

    feature, value = find_split(data)
    left_data = data[data[:, feature] < value]		# Also changed for the node to have value: x<value not x<=value (related to comment in find_split())
    right_data = data[data[:, feature] >= value]

    left_branch, left_depth = decision_tree_learning(left_data, depth+1)
    right_branch, right_depth = decision_tree_learning(right_data, depth+1)

    return {
        "leaf": False,
        "feature": feature,
        "value": value,
        "left": left_branch,
        "right": right_branch
    }, max(left_depth, right_depth)  # did the same thing as above. I am not quite sure about the styling convection here tho - arranged styling convention (my best guess)




# Step 3: Classification & Evaluation
def classify(instance, tree):
    if tree["leaf"]:
        return tree["label"]
    
    if instance[tree["feature"]] < tree["value"]:
        return classify(instance, tree["left"])
    else:
        return classify(instance, tree["right"])


def evaluate(test_data, tree):
    correct = 0
    for instance in test_data:
        prediction = classify(instance[:-1], tree)
        if prediction == instance[-1]:
            correct += 1
    return correct / len(test_data)




# Bonus Part: Tree Visualisation
def plot_tree(tree, x=0, y=0, depth=0, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(20, 10))
        
    if 'leaf' in tree:
        ax.text(x, y, f"Leaf: {tree['leaf']}", bbox=dict(boxstyle="round,pad=0.3", edgecolor="green", facecolor="aliceblue"), ha='center')
        return
    
    ax.text(x, y, f"{tree['attribute']} > {tree['value']}", bbox=dict(boxstyle="round,pad=0.3", edgecolor="blue", facecolor="aliceblue"), ha='center')
    
    # Recursively draw the left and right branches
    left_x, right_x = x - 1/(depth + 1), x + 1/(depth + 1)
    plot_tree(tree['left'], left_x, y-1, depth+1, ax)
    plot_tree(tree['right'], right_x, y-1, depth+1, ax)
    
    # Draw lines connecting nodes
    ax.plot([x, left_x], [y-0.1, y-1+0.1], 'k-')
    ax.plot([x, right_x], [y-0.1, y-1+0.1], 'k-')
    
    ax.axis('off')
    plt.show()

