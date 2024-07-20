# Gazed-at-object-prediction-in-retail-environment
We try to detect what product a shopper looks at in a retail store using single-view RGB images from a CCTV (3rd person) viewpoint. In this project, we have implemented Gaze object localization only. The assumption is that after a successful localization, we can directly use BBox of the detected object and pass through a classification pipeline to get the product category. Taking inspiration from the task similarities between single-stage detectors like [CenterNet](https://arxiv.org/abs/1904.07850) and gaze-following; Here we combine both in a single (stage) model.

## Baeline Gaze Detection Model:
The model presented in [A Modular Multimodal Architecture for Gaze Target Prediction: Application to Privacy-Sensitive Settings](https://openaccess.thecvf.com/content/CVPR2022W/GAZE/papers/Gupta_A_Modular_Multimodal_Architecture_for_Gaze_Target_Prediction_Application_to_CVPRW_2022_paper.pdf) by Anshul Gupta, Samy Tafasca and Jean-Marc Odobez. Many thanks to the authors for their awesome work. [GitHub repo](https://github.com/idiap/multimodal_gaze_target_prediction).

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

### Design choices:
1. Instead of using a Gaussian of fixed size ( std = 3 ) as used generally in gaze following tasks we use dynamic size Gaussian (decided by BBox of the GT gazed at object) like in Centernet.
2. We find that using centerNet regression BBoX sector head and training the model does not work directly out of the box. This can be possible due to the longer training time associated with centerNet. This might present a bottleneck since we do not have a large dataset since we are using GOO Real images only. Instead, we opt to use [TTFNet](https://arxiv.org/pdf/1909.00700), which indeed can be trained in a much shorter time than centerNet.
3. Some changes were also based on empirical data, such as replacing RELUs with GELUs. 

## Results:
### Qualitative Results:
Below are a few images highlighting results obtained where we can visualize (topk = 3) gazed-at-object detection results. The GT image with the gazed-at-object (green) BBoX and gaze heatmap is shown on the left. The prediction results (with red, topk=3 ) BBoX were obtained directly from the TTFNet regression head and predicted gaze heatmap from the baseline gaze detection model output. For the sake of completeness, we include both success and failure cases below:

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

The following results are obtained with these initial configurations:- <br />
Training Dataset = GOO Real Train
topk = 1 <br />
loss function (for baseline model) = l2_loss + dir_loss + att_loss <br />
loss function ( for extended model) = l2_loss + dir_loss + att_loss + ttfnet_hm_loss + ttfnet_wh_loss <br />

We also calculate the prediction accuracy of the models in two ways: 
1. energy aggregation accuracy: The predicted BBoX is selected as one where the predicted heatmap has the maximum energy in the GT BBoX. 
2. BBoX head topk accuracy: Here we predict topk BBoX obtained after processing the results of the BBoX prediction head. In case of topk (n > 1), we perform a simple NMS followed by keeping only BBoX with the score above the threshold score (taken as 0.2 by default). <br />
The changes wrt. baseline config is specified separately in the table.

Model type | Test Datset  | AUC | Min. Distance (in pixels) | Avg. Angular Distance (in degrees) | Max energy accuracy(in %) | BBoX head topk accuracy (in %)
| ------------- | ------------- |------------- |------------- |------------- |------------- | ------------- |
| Baseline | Dense Goo Real | 0.9553 | 0.1160 | 19.0602 | 35.6 | NA
| Extended | Dense Goo Real | 0.9849 | 0.1115 | 18.9836 | 38 | 33.41
| Extended | Sparse Goo Real | 0.9870 | 0.1256 | 22.4442 | 36.23 | 30.72
| Extended [energy] | Dense Goo Real | 0.6280 | 0.1056 | 19.5412 | 43.8 | 34.57 
| Extended [energy] | Sparse Goo Real | 0.6862 | 0.1205 | 23.5692 | 39.5 | 25.51
| Extended [topk=3] | Dense Goo Real | 0.9809 | 0.1060 | 20.0947 | 43.97 | 53.28

where Extended [energy] uses the following loss function for the extended model: l2_loss + dir_loss + att_loss + ttfnet_hm_loss + ttfnet_wh_loss + energy_aggregation_loss and  with topk (n>1) considers the case where all of topk BBoX are used for gazed-at-object localization

From the above, results we draw the following conclusions:
1. Extended model does indeed help in improving the results. So our initial hypothesis of fusing the gaze detection model with SSD detector due to task similarity makes somewhat empirical sense.
2. It is encouraging to see that there is not a large drop in performance while testing in the sparse setting. While the min. distance and avg. distance increases are expected as the objects are now placed far apart. Comparable accuracy in both modes also proves that our model can inherently focus on useful objects/products in a typical retail scene.
3. Adding energy aggregation loss leads to an improvement of ~15% in the BBoX pred accuracy with max energy as expected. Counterintuitively, the AUC values decrease by ~36%. This does not make sense in the first look. I conclude that the AUC metric is not an appropriate metric to evaluate the gazed-at-object localization/classification task. Following the trend in literature, it is mentioned here due to its popular position in legacy and (current) gaze following tasks.
4. The last result with topk = 3 can be slightly misleading as in this we relax our prediction results. This configuration not only would improve our topk accuracy but also increase our false positive rates. Hence just like other object detection tasks, AP and mAP would serve as a more appropriate metrics.

## Conclusion:
This work attempts to check whether single-view RGB offers enough information for successful gazed-at-object localization. Since this problem is ill-defined, we expand the information dimensions available by directly starting from a multi-modal solution. Indeed, using pose and (monocular) depth information along with image data helps to solve some of the ambiguity in the task. Thanks to the authors of [A Modular Multimodal Architecture for Gaze Target Prediction: Application to Privacy-Sensitive Settings](https://openaccess.thecvf.com/content/CVPR2022W/GAZE/papers/Gupta_A_Modular_Multimodal_Architecture_for_Gaze_Target_Prediction_Application_to_CVPRW_2022_paper.pdf) for their awesome work that serves as solid foundation to the current attempt. We feel that works like centerNet would be a natural extension of this task. The amalgamation of the two models is what is presented in this work. Not fully satisfied with the results, we suggest some possible improvements in the following section. 

In the single-view RGB case, the current model could still be extended to opportunistically use the eye data whenever a person's eyes are not occluded in the current frame. Using the GOO-Synth dataset (which is quite large compared to the GOO-Real dataset) could also help in (pre) training a larger model. To deterministically solve the current problem (I believe), we need to solve the problem in a multi-view setting. This would not only alleviate the ambiguity of the problem definition (and thus make it more well-defined) but also solve some issues like occlusion.  The (gazed-at-object) localization pipeline could then be extended and used for other applications like self-checkout. Here, along with tracking the shopper's head (in 3D), we also track the products throughout the retail store. However, at the time of this work (in the first quarter of 2022) we couldn't find any open-source dataset for the task multi-view gazed-at-object localization (or classification) in the retail setting. Our work in the multi-view stereo setting covered [here](https://github.com/Varun-Tandon14/Implementation-of-Cross-View-Tracking-for-Multi-Human-3D-Pose-Estimation-at-over-100-FPS) was a small attempt to solve the problem in MVS.  

## TODO
1. Add the modality extraction script.
2. Provide pre-trained weights for the extended model.

