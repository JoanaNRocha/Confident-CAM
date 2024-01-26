# Confident-CAM: Improving Heat Map Interpretation in Chest X-Ray Image Classification

### Abstract
The integration of explanation techniques promotes the comprehension of a model's output and contributes to its interpretation e.g. by generating heat maps highlighting the most decisive regions for that prediction.
However, there are several drawbacks to the current heat map-generating methods. Probability by itself is not indicative of the model's conviction in a prediction, as it is influenced by multiple factors, such as class imbalance. Consequently, it is possible that a model yields two true positive predictions - one with an accurate explanation map, and the other with an inaccurate one. Current state-of-the-art explanations are not able to distinguish both scenarios and alert the user to dubious explanations. The goal of this work is to represent these maps more intuitively based on how confident the model is regarding the diagnosis, by adding an extra validation step over the state-of-the-art results that indicates whether the user should trust the initial explanation or not. The proposed method, Confident-CAM, facilitates the interpretation of the results by measuring the distance between the output probability and the corresponding class threshold, using a confidence score to generate nearly null maps when the initial explanations are most likely incorrect. This study implements and validates the proposed algorithm on a multi-label chest X-ray classification exercise, targeting 14 radiological findings in the ChestX-Ray14 dataset with significant class imbalance. Results indicate that confidence scores can distinguish likely accurate and inaccurate explanations. 


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

Rocha, Joana, et al. "Confident-CAM: Improving Heat Map Interpretation in Chest X-Ray Image Classification." 2023 IEEE International Conference on Bioinformatics and Biomedicine (BIBM). IEEE, 2023.
https://doi.org/10.1109/BIBM58861.2023.10386065
