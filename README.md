# Using CNN and MLP Neural Networks to recognize faces

## Prerequisites

* install mxnet R package
* download dataset from Kaggle, and strore files in the dataset subdirectory

## run the script to generate a submission file
	
        cd ./R
        Rscript -e "source('main.R') | tee -a output.log"
 
You can select the model to use by adjusting the model name (and parameters) in the main.R file.

Current best score : 3.12247 (rank 36) 
