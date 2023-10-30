Decision Tree Classifier

This coursework presents a decision tree classifier implemented 
from scratch. 
It evaluates the decision tree on a dataset using metrics such 
as the confusion matrix and the related accuracy, precision, 
recall, and F1-score. The tree can be visualized graphically.



1) Running the Code:
	To execute the code and see the results, run the 
	following command (this assumes numpy and matplotlib can be
	imported):

		python main.py

	This command will:
	- Load the clean and noisy datasets.

	- Perform pre-pruning evaluation on both datasets,
	  and print the results in the terminal.

	- Perform post-pruning evaluation on both datasets, 
	  and print the results in the terminal.

	- Train and plot a decision tree on the entire clean 
	  dataset, then display the resulting plot and print 
	  its depth. The plot is also saved in the working 
	  directory as "tree_plot.png".



2) Modifying Datasets Filepaths:
	To use different datasets or paths, you can modify the 
	following lines in the 'if __name__ == "__main__":' section 
	at the end of the code:

		clean_data_filepath = "path_to_the_clean_dataset"
		noisy_data_filepath = "path_to_the_noisy_dataset"

	Replace path_to_the_clean_dataset and 
	path_to_the_noisy_dataset with the desired file paths.



3) Evaluation Methodology:
	The decision tree classifier was evaluated using 
	cross-validation. 
	Both pre and post-pruning evaluations of the datasets 
	were performed using this technique.
	See code docstrings and comments for more details.