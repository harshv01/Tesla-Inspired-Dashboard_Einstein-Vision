import bpy
import pickle
import numpy as np
from mathutils import Vector
import os
import math
import bmesh
import json
import cv2


scene = 'scene13'

def get_center_of_bbox(bbox):
    x1, y1, x2, y2 = bbox
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    return int(center_y), int(center_x)


def check_brake_lights(bb, image_path, red_threshold=0.1):

    image = cv2.imread(image_path)

    cropped_image = image[int(bb[1]):int(bb[3]), int(bb[0]):int(bb[2])]

    hsv_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2HSV)

    lower_red1 = np.array([0, 70, 220])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 70, 220])
    upper_red2 = np.array([180, 255, 255])
    
    mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
    red_mask = mask1 + mask2
    
    red_pixels = np.sum(red_mask > 0)
    total_pixels = cropped_image.shape[0] * cropped_image.shape[1]
    red_percentage = red_pixels / total_pixels

    if red_percentage > red_threshold:
        return True
    else:
        return False



def calculate_luminosity(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    v_channel = hsv[:, :, 2]
    return np.mean(v_channel)

def detect_traffic_light_state(image_path, bbox, brightness_threshold):
    # Load the image
    image = cv2.imread(image_path)

    # Crop the image to the bounding box
    x1, y1, x2, y2 = bbox
    cropped_image = image[int(y1):int(y2), int(x1):int(x2)]

    # Calculate the height of the cropped image and each part (upper, middle, lower)
    height = y2 - y1
    part_height = int(height // 3)

    # Split the cropped image into three parts
    upper_part = cropped_image[:part_height, :]
    middle_part = cropped_image[part_height:2*part_height, :]
    lower_part = cropped_image[2*part_height:, :]  # Adjusted to include the remainder in the lower part

    # Calculate luminosity for each part using the HSV color space
    upper_luminosity = calculate_luminosity(upper_part)
    middle_luminosity = calculate_luminosity(middle_part)
    lower_luminosity = calculate_luminosity(lower_part)

    # Check if all parts are below the brightness threshold
    if upper_luminosity < brightness_threshold and middle_luminosity < brightness_threshold and lower_luminosity < brightness_threshold:
        return "Off"

    # Determine which part has the highest luminosity and is above the threshold
    max_luminosity = max(upper_luminosity, middle_luminosity, lower_luminosity)
    if max_luminosity == upper_luminosity and upper_luminosity >= brightness_threshold:
        return "Red"
    elif max_luminosity == middle_luminosity and middle_luminosity >= brightness_threshold:
        return "Yellow"
    elif max_luminosity == lower_luminosity and lower_luminosity >= brightness_threshold:
        return "Green"
    else:
        return "Off"

def find_closest_point_with_index(given_point, points_list):
    # Initialize minimum distance, the closest point, and the index of the closest point
    min_distance = float('inf')
    closest_point = None
    closest_point_index = -1
    y1, x1 = given_point
    # Iterate through each point in the list along with its index
    for index, point in enumerate(points_list):
        # Unpack the coordinates of the current point
        y2, x2 = point
        
        # Calculate the Euclidean distance
        distance = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
        
        # Update minimum distance, closest point, and index if current distance is smaller
        if distance < min_distance:
            min_distance = distance
            closest_point = point
            closest_point_index = index
            
    return closest_point_index


def get_vehicle_class_orient(point, orient_car_centers, orient_car_orients, class_car_centers=None, class_car_labels=None,get_class=None):
    obj_class = None
    if get_class:
        closest_point_c_index = find_closest_point_with_index(point, class_car_centers)
        obj_class = class_car_labels[closest_point_c_index]
    closest_point_o_index = find_closest_point_with_index(point, orient_car_centers)
    if closest_point_o_index == -1:
        obj_orient = 0
    else:
        obj_orient = orient_car_orients[closest_point_o_index]
    return obj_class, obj_orient


def create_stripes_between_points(point1, point2, width):
    direction = point2 - point1
    perpendicular = Vector((-direction[1], direction[0], 0)).normalized()  # Perpendicular to the line
    
    verts = [
        point1 + width/2 * perpendicular,
        point1 - width/2 * perpendicular,
        point2 - width/2 * perpendicular,
        point2 + width/2 * perpendicular
    ]
    edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
    faces = [(0, 1, 2, 3)]
    mesh = bpy.data.meshes.new("StripesMesh")
    mesh.from_pydata(verts, edges, faces)
    obj = bpy.data.objects.new("Stripes", mesh)
    bpy.context.collection.objects.link(obj)
    return obj

# Define Scale Factor
scale_factor = 1700
y_inrement_for_traffic_lights = 17
cim = np.array([[1600.32128, 0, 637.475995], [0, 1613.79525, 424.541717], [0, 0, 1]])

# Load yolov8 object data
yolo_filepath= f'C:/Users/msult/OneDrive/Desktop/Blender_scripts/pkl_data/yolo_{scene}_all.pkl'
with open(yolo_filepath, 'rb') as file:
    yolo_picle_data = pickle.load(file)

yolo_bbs = yolo_picle_data['bb'] # (y, x) pixels
yolo_labels = yolo_picle_data['labels']

yolo_optic_flow_data = f'C:/Users/msult/OneDrive/Desktop/Blender_scripts/pkl_data/yolo_{scene}_vehicles_optical_flow.pkl'
with open(yolo_optic_flow_data, 'rb') as file:
    yolo_labels_flow = pickle.load(file)

# Load detic object data
detic_filepath= f'C:/Users/msult/OneDrive/Desktop/Blender_scripts/pkl_data/{scene}_all.pkl'
with open(detic_filepath, 'rb') as file:
    detic_picle_data = pickle.load(file)

detic_bbs = detic_picle_data['bb'] # (y, x) pixels
detic_all_imgs_labels_nums = detic_picle_data['labels']

detic_label_names = {5: 'car_(automobile)', 65: 'truck', 87: 'pickup_truck', 55: 'bus_(vehicle)', 34: 'minivan', 0: 'person', 58: 'motorcycle', 46: 'bicycle',
                     40: 'traffic_light', 66: 'cone', 127: 'stop_sign', 44: 'trash_can', 48: 'barrel', 176: 'fireplug', 267: 'speed_limit_sign', 305: 'crosswalk sign'}

optic_flow_data = f'C:/Users/msult/OneDrive/Desktop/Blender_scripts/pkl_data/yolo_{scene}_vehicles_optical_flow.pkl'
with open(optic_flow_data, 'rb') as file:
    all_vehicle_bbs_labels_flow = pickle.load(file)

# Load lanes data
lanes_filepath= f'C:/Users/msult/OneDrive/Desktop/Blender_scripts/pkl_data/{scene}_lanes.pkl'
with open(lanes_filepath, 'rb') as file:
    all_imgs_lane_data = pickle.load(file)  # dicts for each img

# Load car orientation data
car_orient_filepath= f'C:/Users/msult/OneDrive/Desktop/Blender_scripts/pkl_data/{scene}_centers_orients.pkl'
with open(car_orient_filepath, 'rb') as file:
    car_orient_data = pickle.load(file)

orient_car_centers = car_orient_data["centers"]
orient_car_orients = car_orient_data["orientations"]



for i in range(len(detic_all_imgs_labels_nums)):   # loop through each image
#for i in range(0, 5):
    
    # Deselect all objects to start fresh
    bpy.ops.object.select_all(action='DESELECT')

    # Select all objects except the camera and the sun
    for obj in bpy.context.scene.objects:
        if obj.type != 'CAMERA' and obj.name not in ['Sun', 'Plane']:  # Adjust 'Sun' as needed
            obj.select_set(True)

    # Delete the selected objects, leaving the camera and the sun untouched
    bpy.ops.object.delete()

    depths_folder_path = f'C:/Users/msult/OneDrive/Desktop/Blender_scripts/depth_npy/depth_npy_{scene}'
    depth_file_names = os.listdir(depths_folder_path)
    depth_file_names.sort(key=lambda x:int(x.strip('_pred.npy').strip('frame')))
    depth_map = np.load(depths_folder_path + '/' + depth_file_names[i])

    num_human = 0

    for j in range(len(yolo_labels[i])):
        
        label_name = yolo_labels[i][j]
        center = get_center_of_bbox(yolo_bbs[i][j])

        if label_name in ['motorcycle', 'bicycle']:

            _, vehicle_orient = get_vehicle_class_orient(center, orient_car_centers[i], orient_car_orients[i])

            fbx_file_path = f"C:\\Users\\msult\\OneDrive\\Desktop\\Yolov8\\P3Data\\Assets\\{label_name}.fbx"

            bpy.ops.import_scene.fbx(filepath=fbx_file_path)

            imported_object = bpy.context.selected_objects[0]

            x_centre_in_m = -(depth_map[center[0], center[1]] * scale_factor * (center[1] - 637.475995))/1600.32128
            
            location = (depth_map[center[0], center[1]] * scale_factor , x_centre_in_m, 0)
            imported_object.location = location
            imported_object.rotation_euler = (0, 0, -vehicle_orient)


    for bb, [label, status] in yolo_labels_flow[i].items():

        center = get_center_of_bbox(bb)

        _, vehicle_orient = get_vehicle_class_orient(center, orient_car_centers[i], orient_car_orients[i])

        fbx_file_path = f"C:\\Users\\msult\\OneDrive\\Desktop\\Yolov8\\P3Data\\Assets\\{label}_{status}.fbx"
        
        img_path = f'C:\\Users\\msult\\OneDrive\\Desktop\\Yolov8\\P3Data\\Frames\\{scene}\\frame{i + 1}.jpg'
        
        brake_light_status = check_brake_lights(bb, img_path)

        if brake_light_status:
            fbx_file_path = f"C:\\Users\\msult\\OneDrive\\Desktop\\Yolov8\\P3Data\\Assets\\{label}_both_{status}.fbx"

        bpy.ops.import_scene.fbx(filepath=fbx_file_path)

        imported_object = bpy.context.selected_objects[0]

        x_centre_in_m = -(depth_map[center[0], center[1]] * scale_factor * (center[1] - 637.475995))/1600.32128
        
        location = (depth_map[center[0], center[1]] * scale_factor , x_centre_in_m, 0)
        imported_object.location = location
        if label == 'truck':
            imported_object.rotation_euler = (0, 0, -vehicle_orient + np.pi)
        if label == 'bus_(vehicle)':
            imported_object.rotation_euler = (0, 0, -vehicle_orient)
        if label == 'pickup_truck':
            imported_object.rotation_euler = (np.pi/2, 0, -vehicle_orient)
            location = (depth_map[center[0], center[1]] * scale_factor , x_centre_in_m, 15)
            imported_object.location = location
        else:
            imported_object.rotation_euler = (0, 0, -vehicle_orient)
            
    #    elif label_name == 'person':
    #        num_human += 1
    #        human_path = f'C:/Users/msult/OneDrive/Desktop/Blender_scripts/humans/{scene}/frame_{i+1}_{num_human}.obj'
    #        bpy.ops.wm.obj_import(filepath=human_path)
           
    #        imported_object = bpy.context.selected_objects[0]

    #        # Make sure the object is the active object
    #        bpy.context.view_layer.objects.active = imported_object
           
    #        # Set the object's origin to its geometry center
    #        bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
           
    #        # Reset object's rotation to align with global axes
    #        imported_object.rotation_euler = (0, 0, 0)
           
    #        # Set object's location to global origin temporarily
    #        imported_object.location = (0, 0, 0)
           
    #        # Calculate the desired location
    #        x_centre_in_m = -(depth_map[center[0], center[1]] * scale_factor * (center[1] - 637.475995))/1600.32128
    #        location = (depth_map[center[0], center[1]] * scale_factor, x_centre_in_m, 0)
           
    #        # Move the object to the calculated location
    #        imported_object.location = location

    for j in range(len(detic_all_imgs_labels_nums[i])):
        
        if detic_all_imgs_labels_nums[i][j] == 49:
            continue

        label_name = detic_label_names[detic_all_imgs_labels_nums[i][j]]
        center = get_center_of_bbox(detic_bbs[i][j])
        
        if label_name == 'crosswalk sign':
            continue
        
        # if label_name in ['motorcycle', 'bicycle']:

        #     _, vehicle_orient = get_vehicle_class_orient(center, orient_car_centers[i], orient_car_orients[i])

        #     fbx_file_path = f"C:\\Users\\msult\\OneDrive\\Desktop\\Yolov8\\P3Data\\Assets\\{label_name}.fbx"

        #     bpy.ops.import_scene.fbx(filepath=fbx_file_path)

        #     imported_object = bpy.context.selected_objects[0]

        #     x_centre_in_m = -(depth_map[center[0], center[1]] * scale_factor * (center[1] - 637.475995))/1600.32128
            
        #     location = (depth_map[center[0], center[1]] * scale_factor , x_centre_in_m, 0)
        #     imported_object.location = location
        #     imported_object.rotation_euler = (0, 0, -vehicle_orient)
        
        elif label_name in ['stop_sign', 'barrel', 'fireplug', 'cone', 'signboard']:

            fbx_file_path = f"C:\\Users\\msult\\OneDrive\\Desktop\\Yolov8\\P3Data\\Assets\\{label_name}.fbx"

            bpy.ops.import_scene.fbx(filepath=fbx_file_path)

            imported_object = bpy.context.selected_objects[0]
            
            x_centre_in_m = -(depth_map[center[0], center[1]] * scale_factor * (center[1] - 637.475995))/1600.32128
            
            location = (depth_map[center[0], center[1]] * scale_factor , x_centre_in_m, 0)
            if label_name == 'cone':
                imported_object.scale = (5, 5, 5)
            imported_object.location = location
        

        elif label_name == 'trash_can':
            
            obj_file_path = f"C:\\Users\\msult\\OneDrive\\Desktop\\Yolov8\\P3Data\\Assets\\{label_name}.obj"

            bpy.ops.wm.obj_import(filepath=obj_file_path)
            
            imported_object = bpy.context.selected_objects[0]
             
            # Assuming depth_map and scale_factor are defined and contain valid data
            x_centre_in_m = -(depth_map[center[0], center[1]] * scale_factor * (center[1] - 637.475995))/1600.32128
            location = (depth_map[center[0], center[1]] * scale_factor , x_centre_in_m, 0)

            imported_object.location = location
        
        
        elif label_name == 'traffic_light':

            img_path = f"C:\\Users\\msult\\OneDrive\\Desktop\\Yolov8\\P3Data\\Frames\\{scene}\\frame{i + 1}.jpg"

            color = detect_traffic_light_state(img_path, detic_bbs[i][j], brightness_threshold=50)
            
            fbx_file_path = f"C:\\Users\\msult\\OneDrive\\Desktop\\Yolov8\\P3Data\\Assets\\TrafficSignal_{color}.fbx"

            bpy.ops.import_scene.fbx(filepath=fbx_file_path)

            imported_object = bpy.context.selected_objects[0]
            
            x_centre_in_m = -(depth_map[center[0], center[1]] * scale_factor * (center[1] - 637.475995))/1600.32128
            y_centre_in_m = -(depth_map[center[0], center[1]] * scale_factor * (center[0] - 424.541717))/1613.79525 + y_inrement_for_traffic_lights  # + 34 because top of traffic light is origin
            
            location = (depth_map[center[0], center[1]] * scale_factor , x_centre_in_m, y_centre_in_m)
            imported_object.location = location
    
    # for bb, [num_label, status] in all_vehicle_bbs_labels_flow[i].items():
        
    #     label = detic_label_names[num_label]

    #     center = get_center_of_bbox(bb)

    #     _, vehicle_orient = get_vehicle_class_orient(center, orient_car_centers[i], orient_car_orients[i])

    #     fbx_file_path = f"C:\\Users\\msult\\OneDrive\\Desktop\\Yolov8\\P3Data\\Assets\\{label}_{status}.fbx"
        
    #     img_path = f'C:\\Users\\msult\\OneDrive\\Desktop\\Yolov8\\P3Data\\Frames\\{scene}\\frame{i + 1}.jpg'
        
    #     brake_light_status = check_brake_lights(bb, img_path)

    #     if brake_light_status:
    #         fbx_file_path = f"C:\\Users\\msult\\OneDrive\\Desktop\\Yolov8\\P3Data\\Assets\\{label}_both_{status}.fbx"

    #     bpy.ops.import_scene.fbx(filepath=fbx_file_path)

    #     imported_object = bpy.context.selected_objects[0]

    #     x_centre_in_m = -(depth_map[center[0], center[1]] * scale_factor * (center[1] - 637.475995))/1600.32128
        
    #     location = (depth_map[center[0], center[1]] * scale_factor , x_centre_in_m, 0)
    #     imported_object.location = location
    #     if label == 'truck':
    #         imported_object.rotation_euler = (0, 0, -vehicle_orient + np.pi)
    #     if label == 'bus_(vehicle)':
    #         imported_object.rotation_euler = (0, 0, -vehicle_orient)
    #     if label == 'pickup_truck':
    #         imported_object.rotation_euler = (np.pi/2, 0, -vehicle_orient)
    #         location = (depth_map[center[0], center[1]] * scale_factor , x_centre_in_m, 15)
    #         imported_object.location = location
    #     else:
    #         imported_object.rotation_euler = (0, 0, -vehicle_orient)


    for k in range(len(all_imgs_lane_data[i]["dotted-line"])):      # Place stripes
        if len(all_imgs_lane_data[i]["dotted-line"][k]) != 2:
            continue  # Skip pairs with less than 2 points
        pt1, pt2 = all_imgs_lane_data[i]["dotted-line"][k]
        depth1 = depth_map[pt1[0], pt1[1]]
        depth2 = depth_map[pt2[0], pt2[1]]
        
        x_in_m_1 = -(depth1 * scale_factor * (pt1[1] - 637.475995))/1600.32128
        x_in_m_2 = -(depth2 * scale_factor * (pt2[1] - 637.475995))/1600.32128
        
        adjusted_point1 = Vector((depth1 * scale_factor, x_in_m_1, 0))
        adjusted_point2 = Vector((depth2 * scale_factor, x_in_m_2, 0))
        create_stripes_between_points(adjusted_point1, adjusted_point2, width=3)
        
    
        
    arrows = all_imgs_lane_data[i]["arrow"]
        
    for k in range(len(arrows)):
        coordinates = arrows[k]
        mesh = bpy.data.meshes.new("MyMesh")  # Create a new mesh
        obj = bpy.data.objects.new("MyObject", mesh)  # Create a new object linked to the mesh

        # Link the object to the scene collection
        bpy.context.collection.objects.link(obj)

        # Set the new object as active and select it
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)

        # Create a bmesh instance to manipulate the mesh
        bm = bmesh.new()
        
        # Add vertices to the mesh
        for coord in coordinates:
            x_centre_in_m = -(depth_map[coord[0], coord[1]] * scale_factor * (coord[1] - 637.475995)) / 1600.32128
            bm.verts.new((depth_map[coord[0], coord[1]] * scale_factor, x_centre_in_m, 0))

        # Update the bmesh to the mesh
        bm.to_mesh(mesh)
        bm.free()  # Free and update the mesh

    
    for k in range(len(all_imgs_lane_data[i]["solid-line"])):
        
        bpy.ops.curve.primitive_bezier_curve_add()
    
        bez_curve = bpy.context.active_object
            
        for m, pt in enumerate(all_imgs_lane_data[i]["solid-line"][k]):
            
            if m >= len(bez_curve.data.splines[0].bezier_points):
                bez_curve.data.splines[0].bezier_points.add(count=1)

            point = bez_curve.data.splines[0].bezier_points[m]
            point.handle_left_type = 'AUTO'
            point.handle_right_type = 'AUTO'
            
            x_in_m = -(depth_map[pt[0], pt[1]] * scale_factor * (pt[1] - 637.475995))/1600.32128

            point.co = (depth_map[pt[0], pt[1]] * scale_factor, x_in_m, 0)


        bpy.ops.object.convert(target='MESH')

        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.mesh.extrude_region_move(TRANSFORM_OT_translate={"value":(0, 3, 0)})
        bpy.ops.object.mode_set(mode='OBJECT')
    

    bpy.context.scene.render.image_settings.file_format = 'JPEG'
    output_path = f'C:/Users/msult/OneDrive/Desktop/Blender_scripts/render_imgs/{scene}/render{i}.jpg'
    bpy.context.scene.render.filepath = output_path
    bpy.ops.render.render(write_still=True)