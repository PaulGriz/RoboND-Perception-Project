#!/usr/bin/env python

# Import modules
import numpy as np
import sklearn
from sklearn.preprocessing import LabelEncoder
import pickle
from sensor_stick.srv import GetNormals
from sensor_stick.features import compute_color_histograms
from sensor_stick.features import compute_normal_histograms
from visualization_msgs.msg import Marker
from sensor_stick.marker_tools import *
from sensor_stick.msg import DetectedObjectsArray
from sensor_stick.msg import DetectedObject
from sensor_stick.pcl_helper import *

import rospy
import tf
from geometry_msgs.msg import Point
from geometry_msgs.msg import Pose
from std_msgs.msg import Float64
from std_msgs.msg import Int32
from std_msgs.msg import String
from pr2_robot.srv import *
from rospy_message_converter import message_converter
import yaml


def get_normals(cloud):
    """Helper function to get surface normals"""
    get_normals_prox = rospy.ServiceProxy(
        '/feature_extractor/get_normals', GetNormals)
    return get_normals_prox(cloud).cluster


def make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose):
    """Helper function to create a yaml friendly dictionary from ROS messages"""
    yaml_dict = {}
    yaml_dict["test_scene_num"] = test_scene_num.data
    yaml_dict["arm_name"] = arm_name.data
    yaml_dict["object_name"] = object_name.data
    yaml_dict["pick_pose"] = message_converter.convert_ros_message_to_dictionary(
        pick_pose)
    yaml_dict["place_pose"] = message_converter.convert_ros_message_to_dictionary(
        place_pose)
    return yaml_dict


def send_to_yaml(yaml_filename, dict_list):
    """Helper function to output to yaml file"""
    data_dict = {"object_list": dict_list}
    with open(yaml_filename, 'w') as outfile:
        yaml.dump(data_dict, outfile, default_flow_style=False)


def pcl_callback(pcl_msg):
    """"Callback function for your Point Cloud Subscriber.
    Perception Pipeline from Exercises 1, 2, and 3."""

    # ------------------------------------------------------------ #
    #                   Code from Exercise-2:
    # ------------------------------------------------------------ #

    #   1.) Convert ROS msg to PCL data
    # ============================================================ #
    cloud = ros_to_pcl(pcl_msg)

    #   2.) Statistical Outlier Filtering
    # ============================================================ #
    outlier_filter = cloud.make_statistical_outlier_filter()
    # Set the number of neighboring points to analyze for any given point
    outlier_filter.set_mean_k(20)
    # Set threshold scale factor
    x = 0.3
    # Any point with a mean distance larger than global (mean distance+x*std_dev)
    # will be considered outlier
    outlier_filter.set_std_dev_mul_thresh(x)
    outlier_filtered = outlier_filter.filter()

    #   3.) Voxel Grid Downsampling
    # ============================================================ #
    vox = outlier_filtered.make_voxel_grid_filter()
    # Voxel Size: Larger = lower resolution & faster compute times
    # During my tests: 0.0165 was the highest possible value that
    # kept the 100% detection rate in all three scenes.
    LEAF_SIZE = 0.0165
    # Set the voxel size
    vox.set_leaf_size(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE)
    # Filter results from Voxel Grid Downsampling the point cloud
    vox_filtered = vox.filter()

    #   4.) PassThrough Filter
    # ============================================================ #
    # Create a PassThrough filter object for the Z Axis
    passthrough_z = vox_filtered.make_passthrough_filter()
    filter_axis = 'z'
    passthrough_z.set_filter_field_name(filter_axis)
    # Range of value to keep
    z_axis_min = 0.59
    z_axis_max = 1.05
    passthrough_z.set_filter_limits(z_axis_min, z_axis_max)
    # Filter results from PassThrough along the Z Axis
    filtered_passthrough_z = passthrough_z.filter()

    # Create a PassThrough filter object for the Y Axis
    passthrough_y = filtered_passthrough_z.make_passthrough_filter()
    filter_axis = 'y'
    passthrough_y.set_filter_field_name(filter_axis)
    # Range of value to keep
    y_axis_min = -0.425
    y_axis_max = 0.425
    passthrough_y.set_filter_limits(y_axis_min, y_axis_max)
    # Filter results from PassThrough along the Y Axis
    filtered_passthrough_y = passthrough_y.filter()

    #   5.) RANSAC Plane Segmentation
    # ============================================================ #
    # Create an object for segmentation
    ransac_plane = filtered_passthrough_y.make_segmenter()
    # Set the model you wish to fit
    ransac_plane.set_model_type(pcl.SACMODEL_PLANE)
    ransac_plane.set_method_type(pcl.SAC_RANSAC)
    # Max distance for a point to be considered fitting the model
    max_distance = 0.01
    ransac_plane.set_distance_threshold(max_distance)
    # Call the segment function to obtain set of inlier indices and model coefficients
    inliers, coefficients = ransac_plane.segment()

    #   6.) Extract inliers and outliers
    # ============================================================ #
    # Extract inliers
    table = filtered_passthrough_y.extract(inliers, negative=False)
    # Extract outliers
    objects = filtered_passthrough_y.extract(inliers, negative=True)

    #   7.) Euclidean Clustering
    # ============================================================ #
    white_cloud = XYZRGB_to_XYZ(objects)
    tree = white_cloud.make_kdtree()

    # Create object for cluster extraction
    clusters = white_cloud.make_EuclideanClusterExtraction()
    # Set tolerances for distance threshold
    clusters.set_ClusterTolerance(0.0235)
    clusters.set_MinClusterSize(25)
    clusters.set_MaxClusterSize(300)
    # Search the k-d tree for clusters
    clusters.set_SearchMethod(tree)
    # Extract indices for each of the discovered clusters
    cluster_indices = clusters.Extract()

    #   8.) Create Cluster-Mask Point Cloud to color each cluster
    # ============================================================ #
    # Assign a color corresponding to each segmented object in scene
    cluster_color = get_color_list(len(cluster_indices))
    color_cluster_point_list = []

    for j, indices in enumerate(cluster_indices):
        for i, indice in enumerate(indices):
            color_cluster_point_list.append([white_cloud[indice][0],
                                             white_cloud[indice][1],
                                             white_cloud[indice][2],
                                             rgb_to_float(cluster_color[j])])

    # Create new cloud containing all clusters, each with unique color
    clusters_colored = pcl.PointCloud_PointXYZRGB()
    clusters_colored.from_list(color_cluster_point_list)

    #   9.) Convert PCL data to ROS messages
    # ============================================================ #
    ros_outlier_cloud = pcl_to_ros(outlier_filtered)
    ros_vox_cloud = pcl_to_ros(vox_filtered)
    ros_passthrough = pcl_to_ros(filtered_passthrough_z)
    ros_passthrough_y = pcl_to_ros(filtered_passthrough_y)
    ros_clusters_colored = pcl_to_ros(clusters_colored)
    ros_objects = pcl_to_ros(objects)
    ros_table = pcl_to_ros(table)

    #   10.) Publish ROS messages
    # ============================================================ #
    pcl_outlier_pub.publish(ros_outlier_cloud)
    pcl_vox_pub.publish(ros_vox_cloud)
    pcl_passthrough_pub.publish(ros_passthrough)
    pcl_passthrough_pub_y.publish(ros_passthrough_y)
    pcl_objects_pub.publish(ros_objects)
    pcl_table_pub.publish(ros_table)
    pcl_cluster_pub.publish(ros_clusters_colored)

    # ------------------------------------------------------------ #
    #                   Code from Exercise-3:
    # ------------------------------------------------------------ #

    # Classify the clusters and loop through each detected cluster
    detected_objects_labels = []
    detected_objects = []

    for index, pts_list in enumerate(cluster_indices):
        # Grab the points for the cluster
        pcl_cluster = objects.extract(pts_list)

        # Convert the cluster from pcl to ROS using helper function
        ros_cluster = pcl_to_ros(pcl_cluster)

        # Extract histogram features
        # Same process from capture_features.py
        chists = compute_color_histograms(ros_cluster, using_hsv=True)
        normals = get_normals(ros_cluster)
        nhists = compute_normal_histograms(normals)
        feature = np.concatenate((chists, nhists))

        # Make the prediction, retrieve the label for the result
        # and add it to detected_objects_labels list
        prediction = clf.predict(scaler.transform(feature.reshape(1, -1)))
        label = encoder.inverse_transform(prediction)[0]
        detected_objects_labels.append(label)

        # Publish a label into RViz
        label_pos = list(white_cloud[pts_list[0]])
        label_pos[2] += .4
        object_markers_pub.publish(make_label(label, label_pos, index))

        # Add the detected object to the list of detected objects.
        do = DetectedObject()
        do.label = label
        do.cloud = ros_cluster
        detected_objects.append(do)

    rospy.loginfo('Detected {} objects: {}'
                  .format(len(detected_objects_labels), detected_objects_labels))

    # Publish the list of detected objects
    detected_objects_pub.publish(detected_objects)

    # Suggested location for where to invoke the pr2_mover() function within pcl_callback()
    # Add logic to determine whether or not the object detections are robust
    # before calling pr2_mover()
    try:
        pr2_mover(detected_objects)
    except rospy.ROSInterruptException:
        pass


def pr2_mover(object_list):
    """Function to load parameters and request PickPlace service.
    Outputs the .yaml files within the ./output folder."""
    # Variables for .yaml files
    CURRENT_SCENE = Int32()     # Either 1, 2, or 3
    OBJECT_NAME = String()
    WHICH_ARM = String()
    PICK_POSE = Pose()
    PLACE_POSE = Pose()
    DETECTED_OBJECTS_COUNT = 0
    dict_output_list = []       # For writing data to .yaml files

    # Get parameters
    object_list_param = rospy.get_param('/object_list')
    dropbox = rospy.get_param('/dropbox')
    dropbox_green = None
    dropbox_red = None
    for i in range(len(dropbox)):
        if dropbox[i]['group'] == "green":
            dropbox_green = dropbox[i]
        elif dropbox[i]['group'] == "red":
            dropbox_red = dropbox[i]

    # Loop through the pick list
    for i in range(len(object_list_param)):
        obj_name = object_list_param[i]['name']
        object_group = object_list_param[i]['group']
        # Get the PointCloud for a given object and obtain it's centroid
        labels = []
        centroids = []
        for obj in object_list:
            if (obj_name == obj.label):
                labels.append(obj_name)
                points_arr = ros_to_pcl(obj.cloud).to_array()
                centroid = np.mean(points_arr, axis=0)[:3]
                centroid_scalar = [np.asscalar(centroid[0]),
                                   np.asscalar(centroid[1]),
                                   np.asscalar(centroid[2])]
                centroids.append(centroid_scalar)

        for i in range(len(labels)):
            CURRENT_SCENE.data = rospy.get_param('/test_scene_num')
            OBJECT_NAME.data = labels[i]
            # Select required arm for pick and place to dropbox
            if object_group == "green":
                WHICH_ARM.data = "right"
                PLACE_POSE.position.x = dropbox_green["position"][0]
                PLACE_POSE.position.y = dropbox_green["position"][1]
                PLACE_POSE.position.z = dropbox_green["position"][2]
            else:
                WHICH_ARM.data = "left"
                PLACE_POSE.position.x = dropbox_red["position"][0]
                PLACE_POSE.position.y = dropbox_red["position"][1]
                PLACE_POSE.position.z = dropbox_red["position"][2]
            PICK_POSE.position.x = centroids[i][0]
            PICK_POSE.position.y = centroids[i][1]
            PICK_POSE.position.z = centroids[i][2]
            # Append data to output list to be later saved to output file
            yaml_dict = make_yaml_dict(CURRENT_SCENE, WHICH_ARM, OBJECT_NAME, PICK_POSE, PLACE_POSE)
            dict_output_list.append(yaml_dict)
            DETECTED_OBJECTS_COUNT += 1

    # Output data in yaml format and save within the ./output folder
    if DETECTED_OBJECTS_COUNT > 0:
        yaml_filename = "./output/output_" + str(CURRENT_SCENE.data) + ".yaml"
        print("output file name = %s" % yaml_filename)
        send_to_yaml(yaml_filename, dict_output_list)
    print("Success picking up object number = %d" % DETECTED_OBJECTS_COUNT)
    return


if __name__ == '__main__':

    # 1.) ROS node initialization
    # ---------------------------------------- #
    rospy.init_node('pr2', anonymous=True)

    # 2.) Create Subscribers
    # ---------------------------------------- #
    pcl_sub = rospy.Subscriber("/pr2/world/points", pc2.PointCloud2,
                               pcl_callback, queue_size=1)

    # 3.) Create Publishers
    # ---------------------------------------- #
    object_markers_pub = rospy.Publisher("/object_markers", Marker, queue_size=1)
    detected_objects_pub = rospy.Publisher("/detected_objects", DetectedObjectsArray,
                                           queue_size=1)
    pcl_outlier_pub = rospy.Publisher("/pcl_outlier", PointCloud2, queue_size=1)
    pcl_vox_pub = rospy.Publisher("/pcl_vox", PointCloud2, queue_size=1)
    pcl_passthrough_pub = rospy.Publisher("/pcl_pass", PointCloud2, queue_size=1)
    pcl_passthrough_pub_y = rospy.Publisher("/pcl_pass_y", PointCloud2, queue_size=1)
    pcl_objects_pub = rospy.Publisher("/pcl_objects", PointCloud2, queue_size=1)
    pcl_table_pub = rospy.Publisher("/pcl_table", PointCloud2, queue_size=1)
    pcl_cluster_pub = rospy.Publisher("/pcl_cluster", PointCloud2, queue_size=1)
    pcl_collision_pub = rospy.Publisher("/pr2/3D_map/points", PointCloud2, queue_size=1)

    # 4.) Load Model From disk
    # ---------------------------------------- #
    # Training data is located within the ./training folder
    model = pickle.load(open('./training/model.sav', 'rb'))
    clf = model['classifier']
    encoder = LabelEncoder()
    encoder.classes_ = model['classes']
    scaler = model['scaler']

    # 5.) Initialize color_list
    # ---------------------------------------- #
    get_color_list.color_list = []

    # 6.) Spin while node is not shutdown
    # ---------------------------------------- #
    while not rospy.is_shutdown():
        rospy.spin()
