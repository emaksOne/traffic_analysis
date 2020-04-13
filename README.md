![](example.gif)

#### FOLDER STRUCTURE
```
main_folder
│
└─── yolo3 - implementation of yolo v3 ref: https://github.com/qqwweee/keras-yolo3
│
└─── sort - implementation of sort(Simple Object Realtime Traking) ref: https://github.com/abewley/sort
│
└─── load_dataset_toolkit - scripts for downloading train dataset from Open Images Dataset
│
└─── data - folder for videos
│
└─── model_data - folder contains anchors, classes files, annotations etc.
```
#### FLOW
Read video frame by frame and pass it to yolo v3 model. 
Then pass received boxes and scores to multiple object tracking algorithm (SORT) and receive correspondig ids.
When new ids come put its to dictionary with origin position and another info.
Compare current bboxes with origin with the same id. Find out wich objects move in desired direction and wich
cross the line and update corresponding class count.
#### GET STARTED 
from root of the project
```
wget https://pjreddie.com/media/files/yolov3.weights
python convert.py yolov3.cfg yolov3.weights model_data/yolo.h5
```
its download coco weights and convert it to keras friendly form
#### USAGE
```
python video_process.py --input='path/to/input/video' --line='relative position of horizontal line' 
						[--weights='path/to/custom/weights or 'coco'' --output='path/to/output/video' 
						--classes='path/to/classes/file unless use coco']
						
# for example 
python video_process.py --input='data/example2.mp4' --line='0.4' 
```
will use coco pretrained weights by default and then use only car, bus, truck classes
#### TRAIN 
1. Get train dataset and annotations.

	go to load_dataset_toolkit/OIDv4_ToolKit/ and run from this folder
	```
	python main.py downloader --classes 'Van' 'Truck' 'Bus' 'Car' --type_csv train --limit 400
	```
	You can specify different classes and number of images per class(limit).
	Move load_dataset_toolkit/OIDv4_ToolKit/OID/Dataset folder to project root folder (Just want that folder to be in the root).
	From project's root run 
	```
	python load_dataset_toolkit/OIDv4_ToolKit/oid_to_pascal_voc_xml.py
	python load_dataset_toolkit/OIDv4_ToolKit/voc_to_YOLOv3.py
	```

	It will convert origin annotation files to xml and then to txt that reqired for model.
	Move 4_vehicles.txt and 4_vehicles_classes.txt(you can specify this names in voc_to_YOLOv3.py) to model_data folder

2. Train on this data.

	run 
	```
	python train.py --annotations='path/to/annotations' --classes='path/to/classes/file' --anchors='path/to/anchors'
					--logs='path/to/logs/dir/where/model/weights/will/be/saved' --weights='path/to/pretrained/weights'
					[--epochs='number of epochs']

	# for example
	python train.py --annotations='model_data/4_vehicles.txt' --classes='model_data/4_vehicles_classes.txt' --anchors='model_data/yolo_anchors.txt' --logs='logs/001/' --weights='model_data/yolo.h5' --epochs='50'
	```
	it will save weights and another logs in logs/001 folder
