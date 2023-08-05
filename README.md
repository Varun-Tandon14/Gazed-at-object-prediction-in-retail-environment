# Gazed-at-object-prediction-in-retail-environment
We try to detect what product a shopper is looking at in a retail store just using single-view RGB images from a CCTV viewpoint.

## Important points to list:
1. Instead of using a Gaussian of fixed size ( std = 3 ) as used generally in gaze following tasks we use dynamic size Gaussian (decided by BBox of the GT gazed at object)
