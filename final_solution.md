# Student: Aman Bhatta

# Reproduce Results

To reproduce the results instructions are shown in this readme file : [reproduce_results.md](https://github.com/Czajka-Teaching/semester-project-abhatta1234/blob/main/reproduce_results.md)

# Details of the trained model
- Base Architecture: ResNet-50
- Training Dataset: Subset of MS1MV2, with 1M images each for male and female
- Validation Dataset: Kaggle Gender Classification Dataset
- Test Dataset: MORPH

# Accuracy

- Training Accuracy on MS1MV2: 99%
- Test Accuracy on MORPH : 
    - all : 88%
    - Male : 99%
    - Female : 60%
    - Caucasian Male : 99%
    - Caucasian Female: 82%
    - AA_M: 99%
    - AA_F: 50%
- Validation Acurracy on Kaggle Dataset: 95% @ 63rd epoch 

# Results
The gradcam visualizations are split into two categories. The first category of visualizations is where the model's prediction is correct and the second category of visualzations is where the model's prediction is incorrect. The visualization are also split separetly by gender i.e male and female, and also by ethnic groups i.e Caucasian male, Caucasian female, African-American male and African-American female. The visualizations are shown below: 

<table>
    <tr><th colspan=2> Grad-CAM Visualization for Gender Classification Model </th></tr>
  <tr><th colspan=2> CORRECT CLASSIFICATION </th></tr>
  <tr><td align="center"> Average Caucasian Male Face</td><td align="center">Grad-CAM</td></tr>
    <tr><td><img src="/gradcam_images/C_M_average_image_correct.jpg" width="600"/></td><td><img src="/gradcam_images/C_M_gradcam_cam_correct.jpg" width="600"/></tr>
  <tr><td align="center">Average Caucasian Female Face</td><td align="center">Grad-CAM</td></tr>
  <tr><td><img src="/gradcam_images/C_F_average_image_correct.jpg" width="600"/></td><td><img src="/gradcam_images/C_F_gradcam_cam_correct.jpg" width="600"/></td></tr>
    <tr><td align="center">Average African-American Male Face</td><td align="center">Grad-CAM</td></tr>
  <tr><td><img src="/gradcam_images/AA_M_average_image_correct.jpg" width="600"/></td><td><img src="/gradcam_images/AA_M_gradcam_cam_correct.jpg"  width="600"/></td></tr>
  <tr><td align="center">Average Caucasian Male Face</td><td align="center">Grad-CAM</td></tr>
  <tr><td><img src="/gradcam_images/AA_F_average_image_correct.jpg" width="600"/></td><td><img src="/gradcam_images/AA_F_gradcam_cam_correct.jpg" width="600"/></td></tr>
    <tr><td align="center">MORPH Male Face</td><td align="center">Grad-CAM</td></tr>
  <tr><td><img src="/gradcam_images/males_average_image_correct.jpg" width="600"/></td><td><img src="/gradcam_images/males_gradcam_cam_correct.jpg"  width="600"/></td></tr>
  <tr><td align="center">MORPH Female Face</td><td align="center">Grad-CAM</td></tr>
  <tr><td><img src="/gradcam_images/females_average_image_correct.jpg" width="600"/></td><td><img src="/gradcam_images/females_gradcam_cam_correct.jpg" width="600"/></td></tr>
    
  <tr><th colspan=2> INCORRECT CLASSIFICATION </th></tr>
  <tr><td align="center"> Average Caucasian Male Face</td><td align="center">Grad-CAM</td></tr>
    <tr><td><img src="/gradcam_images/C_M_average_image_incorrect.jpg" width="600"/></td><td><img src="/gradcam_images/C_M_gradcam_cam_incorrect.jpg" width="600"/></tr>
  <tr><td align="center">Average Caucasian Female Face</td><td align="center">Grad-CAM</td></tr>
  <tr><td><img src="/gradcam_images/C_F_average_image_incorrect.jpg" width="600"/></td><td><img src="/gradcam_images/C_F_gradcam_cam_incorrect.jpg" width="600"/></td></tr>
    <tr><td align="center">Average African-American Male Face</td><td align="center">Grad-CAM</td></tr>
  <tr><td><img src="/gradcam_images/AA_M_average_image_incorrect.jpg" width="600"/></td><td><img src="/gradcam_images/AA_M_gradcam_cam_incorrect.jpg"  width="600"/></td></tr>
  <tr><td align="center">Average African-American Female Face</td><td align="center">Grad-CAM</td></tr>
  <tr><td><img src="/gradcam_images/AA_F_average_image_incorrect.jpg" width="600"/></td><td><img src="/gradcam_images/AA_F_gradcam_cam_incorrect.jpg" width="600"/></td></tr>
    <tr><td align="center">MORPH Male Face</td><td align="center">Grad-CAM</td></tr>
  <tr><td><img src="/gradcam_images/males_average_image_incorrect.jpg" width="600"/></td><td><img src="/gradcam_images/males_gradcam_cam_incorrect.jpg"  width="600"/></td></tr>
  <tr><td align="center">MORPH Female Face</td><td align="center">Grad-CAM</td></tr>
  <tr><td><img src="/gradcam_images/females_average_image_incorrect.jpg" width="600"/></td><td><img src="/gradcam_images/females_gradcam_cam_incorrect.jpg" width="600"/></td></tr>
</table>
<br><br>

# Analysis/ Discussion

# Trained Model
    
1) The model accuracy varied greatly by gender and also the ethnic group. The test accuracy is near perfect for Caucasian and African-American males. The accuracy accurate for Caucasian female as compared to African-American female, which is almost random.
2) The random accuracy for African-American female might be due to the fact that there was no african-american females in the training dataset. The model is trained on the subset of MS1MV2 celebrity dataset and that is most likely the cause.
3) The other cause might be that the learning rate was reduced based on the plateuing overall validation accuracy. It seems like that optimized for the male accuracy more than for the female accuracy.

# Gradcam images
1) For correct classification,the gradcam images suggest that for the correct classification, the model generally focuses on the lower midregion of the face - mostly on nose and mouth regions. 
2) For incorrect classification, the gradcam activations donot show any conclusive pattern. This simply suggest that the trained model doesn't look at specific region of the face before making incorrect decision. It also suggests that the model looks at different regions of the face for different misclassified images. 
