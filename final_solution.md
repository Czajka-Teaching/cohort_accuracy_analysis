# Details of the trained model
- Base Architecture: ResNet-50
- Training Dataset: Subset of MS1MV2, with 1M images each for male and female
- Validation Dataset: Kaggle Gender Classification Dataset
- Test Dataset: MORPH

# Accuracies

- Train: 99.97
- Test: 
    - all : 88
    - male: 99
    - female:60
    - C_M: 99
    - C_F: 82
    - AA_M: 99
    - AA_F: 50
- Validation: 94.23 @ 63rd epoch 

<table>
    <tr><th colspan=2> Grad-CAM Visualization for Gender Classification Model </th></tr>
  <tr><th colspan=2> Correct Classification </th></tr>
  <tr><td align="center">Average Caucasian Male Face</td><td align="center">Grad-CAM</td></tr>
    <tr><td><img src="/blurPlots/CMcrossmatch/distributionplots/crossmatch_eyes.png" width="600"/></td><td><img src="/blurPlots/CMcrossmatch/distributionplots/crossmatch_brows.png" width="600"/></tr>
  <tr><td align="center">Average Caucasian Male Face</td><td align="center">Grad-CAM</td></tr>
  <tr><td><img src="/blurPlots/CMcrossmatch/distributionplots/crossmatch_mouthwlips.png" width="600"/></td><td><img src="/blurPlots/CMcrossmatch/distributionplots/crossmatch_nose.png" width="600"/></td></tr>
    <tr><td align="center">Average Caucasian Male Face</td><td align="center">Grad-CAM</td></tr>
  <tr><td><img src="/blurPlots/CMcrossmatch/distributionplots/crossmatch_skin.png" width="600"/></td><td><img src="/blurPlots/CMcrossmatch/distributionplots/crossmatch_hair.png" width="600"/></td></tr>
  <tr><td align="center">Average Caucasian Male Face</td><td align="center">Grad-CAM</td></tr>
  <tr><td><img src="/blurPlots/CMcrossmatch/distributionplots/crossmatch_skin.png" width="600"/></td><td><img src="/blurPlots/CMcrossmatch/distributionplots/crossmatch_hair.png" width="600"/></td></tr>
  <tr><th colspan=2> InCorrect Classification </th></tr>
  <tr><td align="center">Average Caucasian Male Face</td><td align="center">Grad-CAM</td></tr>
    <tr><td><img src="/blurPlots/CMcrossmatch/distributionplots/crossmatch_eyes.png" width="600"/></td><td><img src="/blurPlots/CMcrossmatch/distributionplots/crossmatch_brows.png" width="600"/></tr>
  <tr><td align="center">Average Caucasian Male Face</td><td align="center">Grad-CAM</td></tr>
  <tr><td><img src="/blurPlots/CMcrossmatch/distributionplots/crossmatch_mouthwlips.png" width="600"/></td><td><img src="/blurPlots/CMcrossmatch/distributionplots/crossmatch_nose.png" width="600"/></td></tr>
    <tr><td align="center">Average Caucasian Male Face</td><td align="center">Grad-CAM</td></tr>
  <tr><td><img src="/blurPlots/CMcrossmatch/distributionplots/crossmatch_skin.png" width="600"/></td><td><img src="/blurPlots/CMcrossmatch/distributionplots/crossmatch_hair.png" width="600"/></td></tr>
  <tr><td align="center">Average Caucasian Male Face</td><td align="center">Grad-CAM</td></tr>
  <tr><td><img src="/blurPlots/CMcrossmatch/distributionplots/crossmatch_skin.png" width="600"/></td><td><img src="/blurPlots/CMcrossmatch/distributionplots/crossmatch_hair.png" width="600"/></td></tr>
</table>
<br><br>
