# Gazed-at-object-prediction-in-retail-environment
We try to detect what product a shopper is looking at in a retail store just using single-view RGB images from a CCTV viewpoint. In this project, we have implemented Gaze object localization only. After a successful localization, we can directly use BBox of the detected object and pass through a classification pipeline to get the product category. Taking inspiration from the task similarities between single-stage detectors like [CenterNet](https://arxiv.org/abs/1904.07850) and gaze-following; Here we combine both in a single (stage) model.

## Dataset:
### GOO: A Dataset for Gaze Object Prediction in Retail Environments:
* [Paper](https://arxiv.org/abs/2105.10793)
* [Github](https://github.com/upeee/GOO-GAZE2021/tree/main)
* [Donwload information](https://github.com/upeee/GOO-GAZE2021/tree/main/dataset)

### An Image is worth a thousand (16x16?) words:
I think the end goal can be sufficiently explained by ground truth images from the GOO-Real dataset below.
<p>
    <img src="https://github.com/Varun-Tandon14/Gazed-at-object-prediction-in-retail-environment/assets/24519234/514c5c83-5100-4da1-a1d3-842aaeca6ee6" height="300" width="400"/>
    <img src="https://github.com/Varun-Tandon14/Gazed-at-object-prediction-in-retail-environment/assets/24519234/50cfa147-6864-4b60-b265-fd5141296978" height="300" width="400"/>
</p>

## Requirements: 

## Implementation:
### General pipeline:

### Design choices:
1. Instead of using a Gaussian of fixed size ( std = 3 ) as used generally in gaze following tasks we use dynamic size Gaussian (decided by BBox of the GT gazed at object) like in Centernet.

## Results:
