# dtld-yolo-converter

Converts the datasets of the dtld for a yolo format to train a model. This dataset is offered by the Uni-Ulm.

Link to the Dataset: 
- [website](https://www.uni-ulm.de/en/in/institute-of-measurement-control-and-microtechnology/research/data-sets/driveu-traffic-light-dataset/)
- [Github](https://github.com/julimueller/dtld_parsing)

## Using the converter

Install dependencies: 

```
pip3 install ultralytics[all] matplotlib pandas opencv-python
```

First use the ```copy_data.py``` to copy the dtld data for a yolo. In the file, adjust the parameter of ```DATASET_BASE_PATH```
to the dtld. Also set the datasets of cities you want to use. 

## Train

Train the model with the ```train.py```. Adjust again the model parameter inside the file. 
