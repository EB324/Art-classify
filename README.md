# Art-classify

*This program is developed for ISOM5240 course at HKUST.*

This project aims at classifying different styles of art using a multi-layer CNN model. 

## Data description

Data source: https://www.kaggle.com/thedownhill/art-images-drawings-painting-sculpture-engraving 
The original dataset contains 5 categories, from which we choose 3 (iconograohy, painting, sculpture) to build this model. Each category has about 2000 examples. 

## Library requirement

The following libraries are required

* numpy
* matplotlib
* os
* cv2
* tqdm
* keras
* pickle
* sklearn

## File description

* art_classify.py: Main program. Parameters are set according to the results of finetuning. 
* demo.ipynb: An interactive example to show the performance of the program

*The following files are used during developing*

* variables.py: Generate variables from dataset and save them as pickle files. 
* finetune.py: Finetuning parameters.
* model.py: Train model using the best-performed parameters from the finetune process and sava the model as pickle files

All the saved pickle files can be found in the "trained" folder.
