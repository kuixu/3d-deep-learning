# 3d-deep-learning
3D Deep Learning works


## Tasks


### 3D Representation

#### Spherical CNNs
  - Taco S. Cohen, Spherical CNNs, ICLR 2018 Best paper \[[paper](https://openreview.net/forum?id=Hkbd5xZRb)\] 
  - Learning SO\(3\) Equivariant Representations with Spherical CNNs \[[paper](https://arxiv.org/pdf/1711.06721v2.pdf)] [[code](https://github.com/daniilidis-group/spherical-cnn)]
  - Deep Learning Advances on Different 3D Data
Representations: A Survey  \[[paper](https://arxiv.org/pdf/1808.01462.pdf)\] 


### 3D Classification

#### Datasets

  - [ModelNet10/40](http://3dshapenets.cs.princeton.edu)

#### Networks

  - 3D CNN
      - [3D-DenseNet](https://github.com/barrykui/3ddensenet.torch)
      - Voxnet: A 3d convolutional neural network for real-time object recognition, IROS 2015.  \[[code](https://github.com/dimatura/voxnet)\] \[[paper](http://arxiv.org/abs/1505.00880)\]
      - [3D-NIN, network in network]
      - VRN Ensemble, Generative and discriminative voxel modeling with convolutional neural networks, arxiv \[[paper](https://arxiv.org/pdf/1608.04236.pdf)] \[[code](https://github.com/ajbrock/Generative-and-Discriminative-Voxel-Modeling)\]
        - Voxception-Resnet Blocks
  - 2D CNN
        - MVCNN, Learned-Miller.Multi- view convolutional neural networks for 3d shape recognition, ICCV2015 \[[project](http://vis-www.cs.umass.edu/mvcnn/)\] \[[code](https://github.com/suhangpro/mvcnn)\] \[[paper](http://arxiv.org/abs/1505.00880)\]\[[data](http://maxwell.cs.umass.edu/mvcnn-data/)\] \[[video](http://vis-www.cs.umass.edu/mvcnn/docs/1694_video.mp4)\]
  - Point
      - PointNet \[[project](http://stanford.edu/~rqi/pointnet/)]\[[paper](http://arxiv.org/abs/1612.00593)]\[[code](https://github.com/charlesq34/pointnet)]\[[video](https://www.youtube.com/watch?v=Cge-hot0Oc0)][[slides](http://stanford.edu/~rqi/pointnet/docs/cvpr17_pointnet_slides.pdf)]
        - global pooling
        - T-net
      - PointNet++ \[[paper](https://arxiv.org/pdf/1706.02413.pdf)\] \[[code](https://github.com/charlesq34/pointnet2)] 
        - sampling & grouping to learning local feature for fine-gaint objects
        - two PointNet
  - Graph/tree-based 
      - Kd-Net, scape from cells: Deep kd- networks for the recognition of 3d point cloud models, arxiv2017 \[[paper](http://arxiv.org/abs/1704.01222)\]
        - kd-tree
      - Octnet: Learning deep 3d representations at high resolutions, CVPR2017 
        - octree
      - O-cnn: Octree-based convolutional neural networks for 3d shape analysis, TOG2017
        - octree
      - SO-Net: Self-Organizing Network for Point Cloud Analysis, CVPR2018 \[[paper](https://arxiv.org/abs/1803.04249)\]  \[[code](https://github.com/lijx10/SO-Net)\]
        - point-to-node kNN search Self-Organizing Map \(SOM\)  
      - KCNet, Mining Point Cloud Local Structures by Kernel Correlation and Graph Pooling, CVPR2018   \[[project](http://vis-www.cs.umass.edu/mvcnn/)\] \[[code](https://github.com/suhangpro/mvcnn)\] \[[paper](http://arxiv.org/abs/1505.00880)\]\[[data](http://maxwell.cs.umass.edu/mvcnn-data/)\] \[[video](http://vis-www.cs.umass.edu/mvcnn/docs/1694_video.mp4)\]
        - Kernel Correlation
        - Graph Pooling
      - 

### 3D Segmentation

#### Datasets

  - [HVSMR](http://segchd.csail.mit.edu/data.html)
  - [BRATS Data](https://sites.google.com/site/braintumorsegmentation/home/brats2015)
  - [ShapeNet]()

#### Networks

  - HeartSeg, 3D-FC-Densenet, Automatic 3D Cardiovascular MR Segmentation with 
Densely-Connected Volumetric ConvNets - MICCAI 2017 - [[code](https://github.com/yulequan/HeartSeg)]
  - 3D-Unet \[[paper](http://lmb.informatik.uni-freiburg.de/Publications/2016/CABR16/cicek16miccai.pdf)]
  - ClusterNet: 3D Instance Segmentation in RGB-D Images \[[paper](https://arxiv.org/pdf/1807.08894.pdf)\]
  - PointNet \[[project](http://stanford.edu/~rqi/pointnet/)]\[[paper](http://arxiv.org/abs/1612.00593)]\[[code](https://github.com/charlesq34/pointnet)]\[[video](https://www.youtube.com/watch?v=Cge-hot0Oc0)][[slides](http://stanford.edu/~rqi/pointnet/docs/cvpr17_pointnet_slides.pdf)]
  - PointNet++ \[[paper](https://arxiv.org/pdf/1706.02413.pdf)\] \[[code](https://github.com/charlesq34/pointnet2)] 
  - KCNet, Mining Point Cloud Local Structures by Kernel Correlation and Graph Pooling, CVPR2018   \[[project](http://vis-www.cs.umass.edu/mvcnn/)\] \[[code](https://github.com/suhangpro/mvcnn)\] \[[paper](http://arxiv.org/abs/1505.00880)\]\[[data](http://maxwell.cs.umass.edu/mvcnn-data/)\] \[[video](http://vis-www.cs.umass.edu/mvcnn/docs/1694_video.mp4)\]
  - SO-Net: Self-Organizing Network for Point Cloud Analysis, CVPR2018 \[[paper](https://arxiv.org/abs/1803.04249)\]  \[[code](https://github.com/lijx10/SO-Net)\]
  - 3D Shape Segmentation with Projective Convolutional Networks. CVPR2017. [`Project`](http://people.cs.umass.edu/~kalo/papers/shapepfcn/) [`Poster`](http://people.cs.umass.edu/~kalo/papers/shapepfcn/ShapePFCN_poster.pdf) [`Presentation`](http://people.cs.umass.edu/~kalo/papers/shapepfcn/ShapePFCN_poster.pdf) 
  - nnU-Net: Breaking the Spell on Successful Medical Image Segmentation \[[paper](https://arxiv.org/pdf/1904.08128.pdf)\] \[[code](https://github.com/MIC-DKFZ/nnunet)] 

### 3D Object Detection

#### Datasets

Data types: RGBD, Flow, Laser
  - [KITTI](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d)
  - [KITTI Object Visualization Tool](https://github.com/barrykui/kitti_object_vis)

#### Networks  

  - MV3D, Multi-View 3D Object Detection Network for Autonomous Driving \[[paper](https://arxiv.org/pdf/1611.07759)\] [[code](https://github.com/bostondiditeam/MV3D)]
  - Avod, Joint 3D Proposal Generation and Object Detection from View Aggregation \[[paper](https://arxiv.org/abs/1712.02294)\] [[code](https://github.com/kujason/avod)]
  - F-PointNet, Frustum PointNets for 3D Object Detection from RGB-D Data \[[paper](https://arxiv.org/abs/1711.08488)\] \[[code](https://github.com/charlesq34/frustum-pointnets)\]
  - VoxelNet: End-to-End Learning for Point Cloud Based 3D Object Detection \[[paper](https://arxiv.org/abs/1711.06396)\]
  - PIXOR: Real-time 3D Object Detection from Point Clouds - CVPR 2018 -  \[[paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Yang_PIXOR_Real-Time_3D_CVPR_2018_paper.pdf)\] \[[code](https://github.com/charlesq34/frustum-pointnets)\]
  - 3D Object Proposals using Stereo Imagery for Accurate Object Class Detection \[[paper](https://arxiv.org/abs/1608.07711)\]
  - 3D Bounding Box Estimation Using Deep Learning and Geometry \[[paper](https://arxiv.org/abs/1612.00496)\]  \[[code](https://github.com/smallcorgi/3D-Deepbox)\]
  - [Learning 3D Object Orientations From Synthetic Images](http://cs231n.stanford.edu/reports/rqi_final_report.pdf)

### 3D Reconstruction & Generation

#### Datasets

Data types: RGBD, Flow, Laser
  - ShapeNet

#### Networks  

  - SO-Net: Self-Organizing Network for Point Cloud Analysis, CVPR2018 \[[paper](https://arxiv.org/abs/1803.04249)\]  \[[code](https://github.com/lijx10/SO-Net)\]
  - 3D-GAN  \[[paper](https://arxiv.org/abs/1612.00496)\]  \[[code](https://github.com/zck119/3dgan-release)\]
  - Generating 3D Adversarial Point Clouds \[[paper](https://arxiv.org/pdf/1809.07016.pdf)\] 

### 3D Human Pose Estimation

#### Datasets

Data types: RGBD, Flow, Laser
  - ShapeNet

#### Networks  

  - Synthetic Occlusion Data Augmentation -2018 ECCV PoseTrack Challenge - \[[paper](https://arxiv.org/abs/1809.04987)\] \[[code](https://github.com/isarandi/synthetic-occlusion)\]
  - Towards 3D Human Pose Estimation in the Wild: a Weakly-supervised Approach - ICCV 2017  - \[[paper](https://arxiv.org/abs/1809.04987)\] \[[code](https://github.com/xingyizhou/pose-hg-3d)\] \[[code-pytorch](https://github.com/xingyizhou/Pytorch-pose-hg-3d)
  - 3D human pose estimation from depth maps using a deep combination of poses \[[paper](https://arxiv.org/pdf/1807.05389.pdf)\] 


## CVPR2016 Tutorial: 3D Deep Learning with Marvin
  - [CVPR2016 Tutorial: 3D Deep Learning with Marvin](http://vision.princeton.edu/event/cvpr16/3DDeepLearning/)
  - [3D Shape Retrieval](https://shapenet.cs.stanford.edu/shrec16/)
  - [C3D](https://github.com/facebook/C3D), [website](http://www.cs.dartmouth.edu/~dutran/c3d/)
  - [Video Caffe(C3D)] [[code](https://github.com/chuckcho/video-caffe)]
  - [DeepMedic, Brain Lesion Segmentation] [[code(https://github.com/Kamnitsask/deepmedic)]
  - [3D Keypoint Detection and Feature Description](http://staffhome.ecm.uwa.edu.au/~00051632/page100.html)

## Codes and libs for 3D
  - [util3d](https://github.com/fyu/util3d)
  - [spectral-lib](https://github.com/mbhenaff/spectral-lib)
  - [3D-Caffe](https://github.com/yulequan/3D-Caffe#installation)
  - [3D-ResNets-PyTorch](https://github.com/kenshohara/3D-ResNets-PyTorch)


## DL on Medical Image
  - [Antibody-supervised deep learning for quantification of tumor-infiltrating immune cells in hematoxylin and eosin stained breast cancer samples](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5027738/)

## More 3D Papers

see [ [3D-Machine-Learning](https://github.com/timzhang642/3D-Machine-Learning)]
