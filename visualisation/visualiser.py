"""
Visualise the Decision Tree
"""

import matplotlib.pyplot as plt


def plot_tree(tree, y=0, depth=0, axis=None, x_coord_dict=None,
              save_filename='./tree_plot.png'):
    COLORS = ["blue", "green", "red", "cyan", "magenta", "orange", "black",
              "purple", "brown", "gray", "olive"]

    if axis is None:
        fig, axis = plt.subplots(figsize=(25, 10))
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