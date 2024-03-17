![Screenshot from 2024-03-18 02-29-17](https://github.com/fh1m/Track_and_Predict/assets/132839265/f581018e-106d-4ac9-bc31-7ec01570b383)# Track and Predict Using [YOLOv7](https://github.com/WongKinYiu/yolov7) The Fastest Object Detection Algorithm 
### only works for Bottles, hehe

This project leverages a powerful object detection system using YOLOv7 integrated with OpenCV to track a bottle in real-time via a webcam feed. 
It provides dynamic detection, tracking, and Prediction of the bottle's direction and speed relative to a predefined reference point.

## Features
* Real-time video feed acquisition from a webcam.
* Object detection using the pre-trained YOLOv7 model.
* Customizable tracking of a bottle's position within the camera's field of view.
* Direction and speed prediction of the bottle's movement in pixels per second.
* Visual output displayed in an OpenCV window.

## Prerequisites
Before running this project, ensure you have the following installed:

* Python 3.8 or above
* OpenCV library
* PyTorch
* PIL (Python Imaging Library)

Additionally, ensure you have the YOLOv7 model weights placed in the correct directory.

## Usage
To run the bottle tracking system, execute the main.py script:

```console
foo@bar:~$ python -m venv env
foo@bar:~$ pip install -r requirements.txt
foo@bar:~$ python main.py
```

This will activate the webcam and start the object detection and tracking process.
The live video feed will display the tracked bottle with a bounding box.
Direction and speed information and prediction will be shown in the bottom and top corners of the OpenCV window.

# Time for some Real stuff
## Working Examples
-----

![Screenshot from 2024-03-18 02-29-17](https://github.com/fh1m/Track_and_Predict/assets/132839265/6ec7aecc-7cba-46af-ab26-6cab3e76efbf)

![Screenshot from 2024-03-18 02-29-22](https://github.com/fh1m/Track_and_Predict/assets/132839265/ae6511e5-cec2-4b00-ae86-eff02179b1db)

![Screenshot from 2024-03-18 02-29-46](https://github.com/fh1m/Track_and_Predict/assets/132839265/196f610c-3e4b-4dfd-9616-1a38734a9de2)

![Screenshot from 2024-03-18 02-30-11](https://github.com/fh1m/Track_and_Predict/assets/132839265/c5afad37-7b7e-4fcb-992c-cc08804ef706)

