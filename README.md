## Dataset Description
* Source: CelebA Spoof and Zalo challenge datasets.
* Total Images: 94,184.
* Labels: Live and Spoof
* Distribution: Training - 65,928 (70%), Validation - 18,837 (20%), Testing - 9,419 (10%).
## Data Preprocessing
### Initial Steps
  * Utilization of the MTCNN model for creating a facial border box.
  * Expansion of the detected box by 40% followed by cropping the face.
### Preprocessing for Training
* Image Resizing: Adjusted to 224x224 pixels.
* Color Conversion: Images converted to grayscale.
* Scaling: Pixel values normalized within the range [0, 1].
## Fine-Tuning Approach
* Flatten the output of the base model.
* Implement a dropout rate of 0.3 to mitigate overfitting.
* Include a dense layer with 8 units (ReLU activation).
*Finalize with a dense layer (Sigmoid activation) for binary classification.
## Model Parameters
* Loss Function: Binary Crossentropy.
* Optimizer: Adam (Learning Rate: 0.000001, Beta_1: 0.9, Beta_2: 0.999, Epsilon: 1e-07).
* Metric: Accuracy.

## Model Performance
| Model | Epochs | Training Loss/Accuracy | Validation Loss/Accuracy | Testing Accuracy |
| --- | --- | --- | --- | --- |
| MobileNet-V2 | 150 | 0.0041 / 99.75% | 0.0109 / 98.21% | 97.08% |
| ResNet50 | 450 | 0.1807 / 99.79% | 0.4891 / 86.07% | 80.8% |

