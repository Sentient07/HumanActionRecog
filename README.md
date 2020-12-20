# README.md

## Overview

This repository contains code for learning embedding space, where visual feature and linguistic feature corresponding to same objects are brought together. Different functionalities of files provided in this repository

	1. data.py         : Processing dataset for training/validation
	2. layers.py   	   : Contains code for all layers : Visual MLP, Linguistic MLP, Spatial Layer for relationship, etc.
	3. net.py 	   	   : Model, Loss function and Optimizers are built here.
	4. train.py    	   : Trains, Performs evaluation and saves evaluation plot of the model.
	5. evaluate.py 	   : Only evaluation and plotting the performance.
	6. utils.py 	   : Averaging functions for losses.
	7. convert_data.py : For dataset conversion. (Only useful if you have the ActionGenome and want to create the JSON from scratch)

## Setup & Installation

	1. Please install all the required packages from requirements.txt provided herewith.
	2. The dataset is huge, so for this task, we have foregone the effort to download the entire dataset. Rather, please setup the DATA directory as follows.
		2.1 DATA
			|-- Annotation
			|   |-- det_withrel_test.json
			|   |-- det_withrel_train.json
			|   |-- objects_inv.json
			|   `-- relationship_inv.json
			|-- ImageFeatures
			|   |-- object_test.json
			|   |-- object_train.json
			|   |-- object_valid_newf.json
			|   |-- subject_test.json
			|   `-- subject_train.json
			`-- WordFeatures
			    |-- obj_word_feat.json
			    |-- object.json
			    |-- predicate.json
			    |-- rel_word_feat.json
			    |-- subject.json
			    `-- visual_phrase.json

		2.2 All the aforementioned JSON files are provided within the directory ActionGenome. 

		2.3 Annotation contains forward relationship(used for positive samples), inverse relationships, used to get negative samples.

		2.4 ImageFeatures : Separate JSON for frame-wise subject and object image features. Also separated by train/test for speed of execution.

		2.5 WordFeatures :
			2.5.1 Separate JSON for frame-wise subject and object word embeddings.
			2.5.2 Mapping of Object and Relationship to its word vector. Used at test time.

	3. WEIGHTS
		3.1 Please extract the weights corresponding to ActionGenome and put it at a convenient location.


## Running

	1. To train, please execute the following command
		```
		$ python train.py --data_path /path/to/Data/ --save_dir /path/to/save_plots/  --exp_name Reproduce --lr_update 3 --num_workers 32 --plot_pr
		```

	2. To evaluate the model,
		``` 
		$ python evaluate.py --data_path /path/to/Data/ --save_dir /path/to/save_plots/  --num_workers 8 --pt_path /path/to/model_epochX.pth
		```

	3. To perform Qualitative testing,
		```
		$ python test.py --data_path /path/to/Data/ --save_dir ./Results/ --im_dir ./TestImages/ --pt_path /path/to/model_epochX.pth 
		```

## Testing
	1. We perform testing on the test set images, split by us from the subset considered from ActionGenome dataset.

	2. A random sample of these images are already placed in the folder `./TestImages` for convenience.

	3. The result generated are plotted in `./Results/`. 

	4. The prediction are on the top of the image in Red ink and the GT is printed on the bottom of the image in Green ink.

	5. Since we consider a subset of ActionGenome and not the entire dataset, the results are on images and not video.

	6. We use pre-trained image features for prediciton.


Warning :

If you encounter `OSError` of too many open files,
	1) Reduce the num_workers to 0.
	2) If it still persists, reduce batch size
	3) Still, it persists, try `$ sync; echo 3 > /proc/sys/vm/drop_caches` --> might close/kill existing applications

If you encounter `OSError: [Errno 16]` then it's actually not an error, the code is working. (https://stackoverflow.com/questions/55693629/getting-oserror-errno-16-device-or-resource-busy-when-using-tf-keras-model)

