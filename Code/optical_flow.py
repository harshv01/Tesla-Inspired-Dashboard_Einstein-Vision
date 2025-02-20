import pickle
import numpy as np
import ipdb
import cv2


def opticalflow(frame1, frame2, point):

    previous = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame1)

    hsv[...,1] = 255

    next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(previous,next,None, 0.5, 3, 15, 3, 10, 1.2, 0)
    magnitude, angle = cv2.cartToPolar(flow[...,0], flow[...,1])
    hsv[...,0] = angle*180/np.pi/2
    hsv[...,2] = cv2.normalize(magnitude,None,0,255,cv2.NORM_MINMAX)
    y, x = point
    disp = flow[int(y)][int(x)]
    flow_val = np.linalg.norm(disp)

    if flow_val<2:
        status = 'Moving'
    else:
        status = 'Parked'
    
    return status
    

scene = 'scene13'

filepath = f'C:\\Users\\msult\\OneDrive\\Desktop\\Blender_scripts\\pkl_data\\yolo_{scene}_all.pkl'

with open(filepath, 'rb') as file:
    detic_data = pickle.load(file)

all_bbs = detic_data['bb']
all_labels = detic_data['labels']
all_vehicles_list = []

for i in range(len(all_labels)):
    print(i)
    vehicle_dict = {}
    for j in range(len(all_labels[i])):

        label = all_labels[i][j]
        bb = all_bbs[i][j]
        x_min, y_min, x_max, y_max = bb
        center = [int((y_min + y_max)/2), int((x_min + x_max)/2)]

        if i == len(all_labels) - 1:
            vehicle_dict[tuple(bb)] = [label, 'Unknown']
            break
        if label in ['car', 'truck', 'bus']:
            frame1 = cv2.imread(f'C:/Users/msult/OneDrive/Desktop/Yolov8/P3Data/Frames/{scene}/frame{i+1}.jpg')
            frame2 = cv2.imread(f'C:/Users/msult/OneDrive/Desktop/Yolov8/P3Data/Frames/{scene}/frame{i+2}.jpg')
            flow = opticalflow(frame1, frame2, center)
            vehicle_dict[tuple(bb)] = [label, flow]
    
    all_vehicles_list.append(vehicle_dict)



output_filepath = f'C:\\Users\\msult\\OneDrive\\Desktop\\Blender_scripts\\pkl_data\\yolo_{scene}_vehicles_optical_flow.pkl'

with open(output_filepath, 'wb') as file:
    pickle.dump(all_vehicles_list, file)
