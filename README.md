>   ## Project: 3D Perception
>
>   >   **Date:** July 15, 2018
>   >
>   >   **By:** Paul Griz
>   >
>   >   Project Submission for the Udacity Robotics Software Engineer Nanodegree

[![Udacity - Robotics NanoDegree Program](https://s3-us-west-1.amazonaws.com/udacity-robotics/Extra+Images/RoboND_flag.png)](https://www.udacity.com/robotics)

---

## Perception Pipeline: Exercises 1, 2 and 3

>   [Link to repo containing the exercises](https://github.com/udacity/RoboND-Perception-Exercises)

#### VoxelGrid Downsampling Filter

RGB-D cameras provide detailed features and dense point clouds resulting in potentially a multitude of points in per unit volume. Using [Big-O Notation](https://en.wikipedia.org/wiki/Big_O_notation), Computing point cloud data is on a <img src="/tex/e7a2f022962441f2be6dc8e70e837b4a.svg?invert_in_darkmode&sanitize=true" align=middle width=40.78082744999999pt height=24.65753399999998pt/> for <img src="/tex/0a058ac87d682266edd77e8a26d16936.svg?invert_in_darkmode&sanitize=true" align=middle width=164.22386475pt height=22.831056599999986pt/>. So, the time required for processing a point cloud is linear to number of points (resolution). However, increasing the resolution is processing time may not yield any improvements. An optimal situation for this problem is finding the lowest <img src="/tex/f9c4988898e7f532b9f826a75014ed3c.svg?invert_in_darkmode&sanitize=true" align=middle width=14.99998994999999pt height=22.465723500000017pt/> for a given accuracy. To accomplish this, I "down-sampled" the data by applying a VoxelGrid Downsampling Filter to derive a point cloud with the lowest <img src="/tex/f9c4988898e7f532b9f826a75014ed3c.svg?invert_in_darkmode&sanitize=true" align=middle width=14.99998994999999pt height=22.465723500000017pt/> while maintaining the accuracy required in representing the input point cloud.

|  ![](./imgs/VoxelGrid-Downsampling-Filter-Example.png)   |
| :------------------------------------------------------: |
| *Figure from Udacity's "Voxel Grid Downsampling" Lesson* |

The figure above displays a 2D & 3D perspective of an example environment.

-   In 2D perspectives, the term "pixel" short for "picture element," and <img src="/tex/f3a4c59a0f89872effb1566a4d311afa.svg?invert_in_darkmode&sanitize=true" align=middle width=356.43871229999996pt height=24.65753399999998pt/>, where <img src="/tex/86f0ddb93d3975fcd459dfef26ab43a7.svg?invert_in_darkmode&sanitize=true" align=middle width=12.32879834999999pt height=22.465723500000017pt/> as the height & <img src="/tex/db33d0d8dee2bdf17dbc403dd12645fe.svg?invert_in_darkmode&sanitize=true" align=middle width=17.12332379999999pt height=22.465723500000017pt/> as the width.
-   In 3D perspectives, the term "voxel" is short for "volume element," and <img src="/tex/dbb1bbb483849e671abc24676b8f324f.svg?invert_in_darkmode&sanitize=true" align=middle width=238.14885434999994pt height=28.670654099999997pt/>, where <img src="/tex/e584a5339f8940eb58bc2f557ee1ad17.svg?invert_in_darkmode&sanitize=true" align=middle width=12.557115449999989pt height=22.465723500000017pt/> is the depth & <img src="/tex/922a708f649353ba87ef05e9f519f297.svg?invert_in_darkmode&sanitize=true" align=middle width=8.21920935pt height=14.15524440000002pt/> is the size of a single voxel.

A voxel grid filter downsamples the data by increasing the <img src="/tex/922a708f649353ba87ef05e9f519f297.svg?invert_in_darkmode&sanitize=true" align=middle width=8.21920935pt height=14.15524440000002pt/>

```python
vox = outlier_filtered.make_voxel_grid_filter()
LEAF_SIZE = 0.0165
vox.set_leaf_size(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE)
vox_filtered = vox.filter()
```

#### Passthrough Filter

>   [Link to Official Passthrough Filter Example](https://github.com/strawlab/python-pcl/blob/3e04e89169bbe15904a03aae6c76b1f4dc20cca5/examples/official/Filtering/PassThroughFilter.py) from python-pcl

Applying a passthrough filter to a 3D point cloud is similar to cropping a photo.

The passthrough filter crops values outside of an input range. The range is specified for each axis with cut-off values along that axis.

PR2 robot's orientation (in the simulation) only required passthrough filters along the Y and Z axis.

-   Along the <img src="/tex/6e0f9c38d0683024eb53cd03772448f6.svg?invert_in_darkmode&sanitize=true" align=middle width=12.739750649999992pt height=22.465723500000017pt/> axis a range of <img src="/tex/2de6c21be5ccc1184c169b6aa4fa96b9.svg?invert_in_darkmode&sanitize=true" align=middle width=113.24215649999998pt height=21.18721440000001pt/> was applied.
-   Along the <img src="/tex/397ec3672012ed7effdca31c941a64d7.svg?invert_in_darkmode&sanitize=true" align=middle width=10.045686749999991pt height=22.465723500000017pt/> axis a range of <img src="/tex/9d0410c028786c85941141bdc7f212a6.svg?invert_in_darkmode&sanitize=true" align=middle width=84.01830524999998pt height=21.18721440000001pt/> was applied.

```python
# PassThrough Filter to remove the areas on the side of the table
#	(The Y axis)
passthrough_y = filtered_passthrough_z.make_passthrough_filter()
filter_axis = 'y'
passthrough_y.set_filter_field_name(filter_axis)
y_axis_min = -0.425
y_axis_max = 0.425
passthrough_y.set_filter_limits(y_axis_min, y_axis_max)
# Filter results from PassThrough along the Y Axis
filtered_passthrough_y = passthrough_y.filter()

# PassThrough Filter to isolate only the objects on the table surface
#	(The Z axis)
passthrough_z = vox_filtered.make_passthrough_filter()
filter_axis = 'z'
passthrough_z.set_filter_field_name(filter_axis)
z_axis_min = 0.59
z_axis_max = 1.05
passthrough_z.set_filter_limits(z_axis_min, z_axis_max)
# Filter results from PassThrough along the Z Axis
filtered_passthrough_z = passthrough_z.filter()
```

| ![](./imgs/PassThrough-Filter.png) |
| :--------------------------------: |
|    *Passthrough Filter Results*    |

#### RANSAC Plane Segmentation

Simple to how [Regression Analysis](https://en.wikipedia.org/wiki/Regression_analysis) finds the relationships among variables, [Random Sample Consensus (RANSAC)](https://en.wikipedia.org/wiki/Random_sample_consensus) will return a model containing inliers and outliers.

| [![img](https://upload.wikimedia.org/wikipedia/commons/thumb/d/de/Fitted_line.svg/255px-Fitted_line.svg.png)](https://en.wikipedia.org/wiki/File:Fitted_line.svg) |
| :----------------------------------------------------------: |
| *Fitted line with RANSAC; outliers have no influence on the result. Credit: [Wikipedia](https://en.wikipedia.org/wiki/Random_sample_consensus)* |

RANSAC is a powerful tool for isolating/identifying objects in computer vision projects. For this project, RANSAC was used to identify the table.

```python
seg = cloud_filtered.make_segmenter()
seg.set_model_type(pcl.SACMODEL_PLANE)
seg.set_method_type(pcl.SAC_RANSAC)
max_distance = 0.01
seg.set_distance_threshold(max_distance)
inliers, coefficients = seg.segment()

# Table
cloud_table = cloud_filtered.extract(inliers, negative=False)
# Objects
cloud_objects = cloud_filtered.extract(inliers, negative=True)
```

The extracted inliers includes the table. It looks like this:

| ![](./imgs/RANSAC.png) |
| :--------------------: |
|    *RANSAC Results*    |

#### Euclidean Clustering

With the table's point cloud now separated from the objects, the objects need to be individually separated. To separate each object, the **Density-based spatial clustering of applications with noise** [(DBSCAN)](https://en.wikipedia.org/wiki/DBSCAN) clustering algorithm was used.

| [![img](https://upload.wikimedia.org/wikipedia/commons/thumb/a/af/DBSCAN-Illustration.svg/400px-DBSCAN-Illustration.svg.png)](https://en.wikipedia.org/wiki/File:DBSCAN-Illustration.svg) |
| :----------------------------------------------------------: |
| *Credit: [Wikipedia](https://en.wikipedia.org/wiki/DBSCAN)*  |

DBSCAN forms clusters by grouping points within an input threshold distance from said point's nearest neighbors.

```python
clusters = white_cloud.make_EuclideanClusterExtraction()
# Set tolerances for distance threshold
clusters.set_ClusterTolerance(0.0235)
clusters.set_MinClusterSize(25)
clusters.set_MaxClusterSize(300)
# Search the k-d tree for clusters
clusters.set_SearchMethod(tree)
# Extract indices for each of the discovered clusters
cluster_indices = clusters.Extract()
```

|         ![](.\imgs\DBSCAN.png)          |
| :-------------------------------------: |
| *DBSCAN / Euclidean Clustering Results* |

#### Object Recognition

TODO FILE SPELL CHECK: To accomplish object recognition, I edited the `./training/capture_features.py` to capture features of each object from `160` random positions each.

The features (output from `capture_features.py`) were then used to train a Support Vector Machine [(SVM)](https://en.wikipedia.org/wiki/Support_vector_machine) to characterize the parameter space of the features into named objects. The SVM uses the [`svm.LinearSVC`](http://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html#sklearn.svm.LinearSVC) Class from the [scikit-learn](http://scikit-learn.org/stable/index.html) Python library.

| ![](.\imgs\Training-Results.png) |
| :------------------------------: |
|        *Training Results*        |

---

### Running the Simulation

1.  Open two terminals.
2.  In both terminals: `cd ~/catkin_ws/src/RoboND-Perception-Project/pr2_robot `
3.  In the first terminal: `roslaunch pr2_robot pick_place_project.launch test_scene_num:=1 `
    1.  *Note:* `test_scene_num` will load scenes [1, 2, or 3].
    2.  Example for scene 3:  `roslaunch pr2_robot pick_place_project.launch test_scene_num:=3`
4.  Wait for `gazibo` & `Rviz` to finish loading.
5.  In the second terminal: `rosrun pr2_robot project.py `
    1.  *Note:* Make sure you are within in the correct directory (see step 2).

---

### Results

| ![](.\imgs\World-1-Results.png) |
| :-----------------------------: |
|           **Scene 1**           |
| Results: 3<img src="/tex/87f05cbf93b3fa867c09609490a35c99.svg?invert_in_darkmode&sanitize=true" align=middle width=8.219209349999991pt height=24.65753399999998pt/>3 Objects Detected |

| ![](.\imgs\World-2-Results.png) |
| :-----------------------------: |
|           **Scene 2**           |
| Results: 5<img src="/tex/87f05cbf93b3fa867c09609490a35c99.svg?invert_in_darkmode&sanitize=true" align=middle width=8.219209349999991pt height=24.65753399999998pt/>5 Objects Detected |

| ![](./imgs/World-3-Results.png) |
| :-----------------------------: |
|           **Scene 3**           |
| Results: 8<img src="/tex/87f05cbf93b3fa867c09609490a35c99.svg?invert_in_darkmode&sanitize=true" align=middle width=8.219209349999991pt height=24.65753399999998pt/>8 Objects Detected |

**Note:** The `.yaml` output files for scenes 1, 2, and 3 are within the `output` directory.

---

### Possible improvements:

1.  Complete the collision mapping challenge by using the `pick_place_server` to execute the pick and place operation.
2.  Further Improve the SVM's training accuracy.
3.  Divide operates in the perception pipeline into separate methods to effectively time and improve operations that require excess computation.
