"""
Visualise the Decision Tree
"""

import matplotlib.pyplot as plt


def plot_tree(tree, y=0, depth=0, axis=None, x_coord_dict=None,
              save_filename='./tree_plot.png'):
    """ Plot the decision tree recursively using Matplotlib.

    Args:
        tree (dict): A dictionary representation of the decision tree node.
                     See `decision_tree_learning` in tree.tree_builder
                     for structure details.
        y (int, optional): The y-coordinate for plotting. Defaults to 0.
        depth (int, optional): Current depth of the tree. Defaults to 0.
        axis (matplotlib axis object, optional): Axis for plotting. Defaults
                                                 to None.
        x_coord_dict (dict, optional): Dictionary to keep track of x-coord.
                                       Defaults to None.
        save_filename (str, optional): Filename for saving the plot. Defaults
                                       to './tree_plot.png'.

    Returns:
        float: The x-coordinate of the plotted node.
    """
    # Define colors for different depths of the tree.
    colours = ["blue", "green", "red", "cyan", "magenta", "orange", "black",
               "purple", "brown", "gray", "olive"]

    # If axis is not provided, initialize a new plot.
    if axis is None:
        fig, axis = plt.subplots(figsize=(25, 15))
        axis.axis('off')
        x_coord_dict = {"next_x": 0}  # Initialise x-coordinates dictionary

    # Determine the color based on the depth of the tree.
    colour = colours[depth % len(colours)]

    # If it's a leaf node, plot the leaf and return its x-coordinate.
    if tree["leaf"]:
        x = x_coord_dict["next_x"]
        x_coord_dict["next_x"] += 1
        axis.text(
            x, y, f"Room:\n{int(tree['label'])}",
            bbox=dict(boxstyle="circle,pad=0.3", edgecolor=colour,
                      facecolor="aliceblue"),
            ha='center', fontsize=6
        )
        return x

    # Recursive calls--Post-order traversal: process children first.
    left_x = plot_tree(tree["left"], y - 1, depth + 1, axis, x_coord_dict)
    right_x = plot_tree(tree["right"], y - 1, depth + 1, axis, x_coord_dict)

    # Determine the x-coordinate for the parent based on the x-coordinates
    # of its children.
    x = (left_x + right_x) / 2.0
    axis.text(
        x, y, f"WiFi_{tree['feature'] + 1} < {tree['value']}",
        bbox=dict(boxstyle="round,pad=0.3", edgecolor=colour,
                  facecolor="aliceblue"),
        ha='center', fontsize=6
    )

    # Draw lines connecting parent and children.
    axis.plot([x, left_x], [y - 0.1, y - 1 + 0.1], colour)
    axis.plot([x, right_x], [y - 0.1, y - 1 + 0.1], colour)

    # If it's the root node, save and display the plot.
    if depth == 0:
        plt.savefig(save_filename, dpi=300)  # Save with high resolution
        plt.show()

    return x