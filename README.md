[Screencast from 2024-03-18 02-32-32.webm](https://github.com/fh1m/Track_and_Predict/assets/132839265/8f960dcc-30e6-4f00-9760-e5c9704521ee)
# Track and Predict Using [YOLOv7](https://github.com/WongKinYiu/yolov7) The Fastest Object Detection Algorithm 
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
foo@bar:~$ source env/bin/activate
foo@bar:~$ python main.py
```

This will activate the webcam and start the object detection and tracking process.
The live video feed will display the tracked bottle with a bounding box.
Direction and speed information and prediction will be shown in the bottom and top corners of the OpenCV window.

# Time for some Real stuff
## Working Examples

![Screenshot from 2024-03-18 02-29-17](https://github.com/fh1m/Track_and_Predict/assets/132839265/6ec7aecc-7cba-46af-ab26-6cab3e76efbf)

![Screenshot from 2024-03-18 02-29-28](https://github.com/fh1m/Track_and_Predict/assets/132839265/5463cf25-e159-4da0-b799-94283fc6ebd7)

### Tested in Different lighting conditions

![Screenshot from 2024-03-18 02-29-46](https://github.com/fh1m/Track_and_Predict/assets/132839265/196f610c-3e4b-4dfd-9616-1a38734a9de2)

![Screenshot from 2024-03-18 02-29-56](https://github.com/fh1m/Track_and_Predict/assets/132839265/ddf972e9-ced6-495d-99d1-3f51432b6db9)


