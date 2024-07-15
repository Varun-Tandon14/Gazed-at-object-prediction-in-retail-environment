# Gazed-at-object-prediction-in-retail-environment
We try to detect what product a shopper looks at in a retail store using single-view RGB images from a CCTV (3rd person) viewpoint. In this project, we have implemented Gaze object localization only. The assumption is that after a successful localization, we can directly use BBox of the detected object and pass through a classification pipeline to get the product category. Taking inspiration from the task similarities between single-stage detectors like [CenterNet](https://arxiv.org/abs/1904.07850) and gaze-following; Here we combine both in a single (stage) model.

## Dataset:
At the time of the implementation, the most relevant dataset for this particular task is the GOO dataset. The dataset contains real (9,552 samples) and synthetic (192000 samples) images. More details about the dataset can be found below- 
### GOO: A Dataset for Gaze Object Prediction in Retail Environments:
* [Paper](https://arxiv.org/abs/2105.10793)
* [Github](https://github.com/upeee/GOO-GAZE2021/tree/main)
* [Download information](https://github.com/upeee/GOO-GAZE2021/tree/main/dataset)

### An Image is worth a thousand (16x16?) words:
Below, ground truth images from the GOO-Real dataset can sufficiently explain the end goal. The target product is highlighted in a green BBox with a Gaussian heatmap as the gaze localization problem is traditionally solved as a regression problem. 
<p>
    <img src="https://github.com/Varun-Tandon14/Gazed-at-object-prediction-in-retail-environment/assets/24519234/514c5c83-5100-4da1-a1d3-842aaeca6ee6" height="300" width="400"/>
    <img src="https://github.com/Varun-Tandon14/Gazed-at-object-prediction-in-retail-environment/assets/24519234/50cfa147-6864-4b60-b265-fd5141296978" height="300" width="400"/>
</p>

## Requirements: 

1. torch
2. torchvision
3. numpy 
4. pandas 
5. cv2
6. matplotlib 
7. csv
8. PIL 
9. datetime
10. timm: >= 0.6.13

## Implementation:
### General pipeline:

### Design choices:
1. Instead of using a Gaussian of fixed size ( std = 3 ) as used generally in gaze following tasks we use dynamic size Gaussian (decided by BBox of the GT gazed at object) like in Centernet.

## Results:
### Qualitative Results:
Below are a few images highlighting results obtained where we can visualize (topk = 3) gazed-at-object detection results. The GT image with the gazed-at-object (green) BBoX and gaze heatmap is shown on the left. The prediction results (with red, topk=3 ) BBoX obtained directly from the ttfnet head and predicted gaze heatmap from the baseline gaze detection model output. For sake of completeness, we include both suceess and failure cases below:

![GOP_result_2](https://github.com/user-attachments/assets/1149e681-b29e-49ec-9628-b03226ddbc65)
![GOP_result_1](https://github.com/user-attachments/assets/521bda82-85a9-4354-953d-cbb520133326)
![GOP_result_16](https://github.com/user-attachments/assets/e47d23c9-55c8-4ead-8b36-711ed0ee1567)
![GOP_result_15](https://github.com/user-attachments/assets/a46772ff-8a62-442e-8e4f-1b63193e064e)
![GOP_result_14](https://github.com/user-attachments/assets/e11e9d06-6803-4f3c-8bfc-cc64e747beb4)
![GOP_result_13](https://github.com/user-attachments/assets/553f8f34-384d-450f-b854-26b96fd7f36b)
![GOP_result_12](https://github.com/user-attachments/assets/024f1ec1-2e61-4e6e-b931-690af4d2d14b)
![GOP_result_11](https://github.com/user-attachments/assets/406f627d-773d-4235-b4cd-065acf6cc7f4)
![GOP_result_10](https://github.com/user-attachments/assets/74abafb7-8af3-4999-86a9-76e2fa6919c3)
![GOP_result_9](https://github.com/user-attachments/assets/b7913c9b-4d2b-4095-b486-dc1cd3a37966)
![GOP_result_8](https://github.com/user-attachments/assets/49f4ddc6-7ccf-4ce5-8754-84d27ad49791)
![GOP_result_7](https://github.com/user-attachments/assets/d1ac4653-b187-409e-a24f-eb3691a56ede)
![GOP_result_6](https://github.com/user-attachments/assets/1757bee8-2a76-4d6d-a826-7ebfa659cd35)
![GOP_result_5](https://github.com/user-attachments/assets/c1ac7131-baea-459f-90cf-b427e5b27718)
![GOP_result_4](https://github.com/user-attachments/assets/79b99c9a-59e3-4e0f-ae3c-582f0743510a)
![GOP_result_3](https://github.com/user-attachments/assets/71fc6066-aa79-4d30-a3d5-8256c765ad5d)

### Quantitative Results: 
