# Confident-CAM: Improving Heat Map Interpretation in Chest X-Ray Image Classification

### Abstract
The integration of explanation techniques allows the assessment of model performance, in order to comprehend its output and further validate it. However, there are several drawbacks to the current heat map-generating methods, namely their low resolution and normalization process, which hinder their comparison and accuracy evaluation. As probability by itself is not indicative of the model's certainty in that class outcome, two positive predictions can be correctly classified but yield more or less accurate visual explanations. Current state-of-the-art explanations are not able to distinguish both scenarios and alert the user to uncertain predictions. The goal of this work is to represent these maps more intuitively based on how certain the model is regarding the diagnosis, by adding an extra validation step over the state-of-the-art results that indicates whether the user should trust the initial explanation or not. More specifically, Confident-CAM proposes to facilitate the interpretation of the results by measuring the distance between the output probability and the corresponding class threshold with a confidence score, to yield nearly null maps when the initial explanations are most likely incorrect. By doing so, this method distinguishes likely accurate and inaccurate explanations, in the last case due to incorrect class predictions or lesion localization. This study implements and validates the proposed methodology on a multi-label chest X-ray classification exercise, targeting the 14 common radiological findings in the ChestX-Ray14 dataset with significant class imbalance.

| ![Figure 2023-05-08 100200](https://github.com/JoanaNRocha/Confident-CAM/assets/44504059/9feccace-03b9-49f1-b803-3a8c038d287a) | 
|:--:| 
| *Example of a false positive "effusion" prediction.* |


### How to Use

#### Requirements
This code was developed using a Pytorch framework. The file with all the requirements is included in the repository (*requirements_env.yml*).

#### File Structure
*ConfidentCAM.py* - Functions for calculating the confidence score and Confident-CAM per instance, per class.

*utils.py* - Multi-label model class, and inference function. 

Data - example input image (*00000001_000.png*), pre-trained multi-label model weights (*model_resnet50.pth*), and corresponding model metrics per class (*test_metrics_resnet50.xlsx*).

#### Example


### Credits
If you use this code, please cite the following publication: 

WIP
