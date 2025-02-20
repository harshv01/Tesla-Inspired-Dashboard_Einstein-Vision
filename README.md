# Tesla Inspired Dashboard (Einstein Vision)

We built a visualization of car's surroundings inspired by that of Tesla's dashboard. We used a set of videos recorded from our professor's Tesla Model S. The visualizations were rendered in Blender.

![crop-1-video-ezgif com-optimize](https://github.com/user-attachments/assets/247e5d71-9b21-4581-aac3-5ae555ce38cd)


## Packages used:

1. **Lane Detection**: mask RCNN (https://debuggercafe.com/lane-detection-using-mask-rcnn/):
	The model gives the masks and bounding boxes of three major classes- 'dotted-line', 'solid-line' and 'divider-line'. Upon the calculation of masks, we find 5 spline curve points for each solid-line and the divider-line, and only two endpoints for each dotted line (each stripe of the striped lane). Using these guide points, lanes are plotted in Blender.
	
2. **Cars, Pedestrians, Stop Signs, Traffic Signals**: YOLOv8 (https://github.com/ultralytics/ultralytics):
	The popular model detects most of our required classes of objects in the form of bounding boxes. The bounding boxes are used to place the mentioned objects in the Blender scene. ZoeDepth is used to figure out the depth (distance from the camera) for better placement.

3. **Car Orientation and Bounding Boxes**: YOLO3d (https://github.com/ruhyadi/YOLO3D):
	Based on the popular YOLO models, YOLO3D provides us with the orientation of the car detected on the frame. The 3D bounding boxes can be used to find all the three axes of rotation, we used only the YAW to represent the cars in our scene.
	
4. **Depth Estimation**: Marigold (https://github.com/prs-eth/Marigold):
	Marigold provides depth value of each pixel in metric form, unlike MiDaS which provides the same in relative form. We used these depths to place out object models in Blender.

5. **Vehicle Classification**: Detic and YOLOv8 (https://github.com/facebookresearch/Detic & https://github.com/ultralytics/ultralytics):
	Detic detects various vehicle types present on the roads like Car, SUV, Truck, bicycle, etc. We use this data to load the appropriate blender model in the frame per scene. The detection between Cars and Trucks was better in YOLOv8, hence we relied on it for the two categories only.

6. **Miscellaneous Objects (trashcans, etc)**: Detic (https://github.com/facebookresearch/Detic):
	In order to detect various other objects on the roads apart from lanes and vehicles, we used Facebook's Detic model. It has 20,000 classes, which proved to be more exhaustive than the COCO datasets used on YOLO models.

7. **Speed Limit Sign Classification**: speed-limit-sign-detection (https://github.com/michaelgallacher/speed-limit-sign-detection):
	This lightweight model detects and classifies specifically the speed limit signs on the roads. 

8. **Human Pose Detection** I2L-MESHNET (https://github.com/mks0601/I2L-MeshNet_RELEASE):
	Generates .obj files matching the pose of the persons detected in the image. The input needs to be in the form of bounding boxes, as the model is unable to detect humans from a large image.

## How to run the code:

1. Install the aforementioned models along with their dependencies in the form of Conda enviroments.
2. Generate the data in the pickle form format as required in the 'blender_code.py'.
3. Place the generated data in the directories as mentioned in 'blender_code.py'.
4. Specify the scene and run blender code.
