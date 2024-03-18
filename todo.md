#### Final Ojective:  Use webcam to track a bottle in real-time and then say where is the bottle is moving based on a reference point, for example if I move the bottle left from the reference point I want the live feed say the bottle is moving left and same for right,backward,forward. also predict the speed and direction of the bottle when its moving(in px)

* get live data feed from webcam [x]
* select pretrained detection model(YOLOv7) & understand why that model works [x]
    1. implement my own custom ML model from sratch [ ]
* Run the detection model(YOLOv7) on each frame captured from the webcam to identify the bottle's position. [x]
* Establish a reference point or area in frame. [x]
* Tracking Movement of the bottle [x]
* Displaying the final Output in opencv box [x]
