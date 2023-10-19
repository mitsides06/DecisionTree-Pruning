# Step 1: Loading data
import numpy as np

data_clean = np.loadtxt("WIFI_db/clean_dataset.txt")
data_noisy = np.loadtxt("WIFI_db/noisy_dataset.txt")


# Step 2: Creating Decision Trees
def entropy(data):
    labels = data[:, -1]
    _, counts = np.unique(labels, return_counts=True)
    probs = counts / len(labels)
    return -np.sum(probs * np.log2(probs))


def gain(s_all, s_left, s_right):
    return entropy(s_all) - (len(s_left)/len(s_all)*entropy(s_left) + len(s_right)/len(s_all)*entropy(s_right))


def find_split(data):
    best_gain = 0
    best_split = None

    for feature in range(data.shape[1] - 1):
        unique_values = np.unique(data[:, feature])
        for value in unique_values:  # we don't need to go over the last unique value as the right split will be empty
            left_split = data[data[:, feature] <= value]
            right_split = data[data[:, feature] > value]

            current_gain = gain(data, left_split, right_split)

            if current_gain > best_gain:
                best_gain = current_gain
                best_split = (feature, value)

    return best_split


def decision_tree_learning(data, depth=0):
    if len(np.unique(data[:, -1])) == 1:
        return ({"leaf": True, "class": data[0, -1]}, depth)  # removed "depth" from the dictionary, and put it as an element in a 2-tuple as our function should return (node, depth)

    feature, value = find_split(data)
    left_data = data[data[:, feature] <= value]
    right_data = data[data[:, feature] > value]

    left_branch, left_depth = decision_tree_learning(left_data, depth+1)
    right_branch, right_depth = decision_tree_learning(right_data, depth+1)

    return ({
        "leaf": False,
        "feature": feature,
        "value": value,
        "left": left_branch,
        "right": right_branch
        }, 
        max(left_depth, right_depth)
        )  # did the same thing as above. I am not quite sure about the styling convection here tho


# Step 3: Evaluation
def classify(instance, tree):
    if tree["leaf"]:
        return tree["class"]
    
    if instance[tree["feature"]] <= tree["value"]:
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

