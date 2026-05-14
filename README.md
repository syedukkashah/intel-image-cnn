# Intel Image Classification using CNN

This is an educational deep learning project where I trained a custom Convolutional Neural Network (CNN) using PyTorch to classify natural scene images into six categories:

- Buildings
- Forest
- Glacier
- Mountain
- Sea
- Street

The purpose of this project was to practice the complete image classification workflow, including data preprocessing, augmentation, CNN model building, training, validation, early stopping, model saving, evaluation, and visualization of results.

---

## Project Overview

In this project, a CNN was trained on the Intel Image Classification dataset. The model learns to classify images into one of six scene categories.

The workflow includes:

1. Loading and exploring the dataset
2. Applying image preprocessing and augmentation
3. Splitting training data into training and validation sets
4. Building a CNN model in PyTorch
5. Training the model using CrossEntropyLoss and Adam optimizer
6. Using a learning rate scheduler and early stopping
7. Saving the best model based on validation loss
8. Evaluating the saved model on the test set
9. Plotting training and validation metrics
10. Creating a confusion matrix and classification report

---

## Dataset

The dataset used in this project is the Intel Image Classification dataset.

[Intel Image Classification Dataset](https://www.kaggle.com/datasets/puneet6060/intel-image-classification/data)

After downloading, place the dataset inside a folder named `data/` with this structure:

```text
data/
├── seg_train/
│   ├── buildings/
│   ├── forest/
│   ├── glacier/
│   ├── mountain/
│   ├── sea/
│   └── street/
│
└── seg_test/
    ├── buildings/
    ├── forest/
    ├── glacier/
    ├── mountain/
    ├── sea/
    └── street/
```

---

## Dataset Class Distribution

The original training set was split into:

- 85% training data
- 15% validation data

---

## Model Architecture

A custom CNN model was created using PyTorch.

The model contains:

- 3 convolutional layers
- ReLU activation functions
- Max pooling layers
- Fully connected layers
- Dropout layers for regularization
- Final output layer with 6 classes

Model summary:

```text
Input Image: 150 x 150 x 3

Conv2d: 3 → 32
ReLU
MaxPool2d

Conv2d: 32 → 64
ReLU
MaxPool2d

Conv2d: 64 → 128
ReLU
MaxPool2d

Flatten

Linear: 18*18*128 → 256
ReLU
Dropout(0.5)

Linear: 256 → 128
ReLU
Dropout(0.4)

Linear: 128 → 6
```

The final layer outputs 6 values, one for each image class.

---

## Training Setup

| Component | Value |
|---|---|
| Framework | PyTorch |
| Loss Function | CrossEntropyLoss |
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Batch Size | 32 |
| Maximum Epochs | 50 |
| Early Stopping Patience | 7 |
| LR Scheduler | ReduceLROnPlateau |
| Model Selection | Lowest validation loss |

The model was trained for a maximum of 50 epochs. Early stopping was used to stop training when validation loss did not improve for 7 consecutive epochs.

The best model was saved as:

```text
best_model.pt
```

---

## Results

Early stopping was triggered at epoch 35.

The best validation loss occurred at epoch 28.

| Metric | Value |
|---|---:|
| Best Validation Loss | 0.4200 |
| Validation Accuracy at Best Loss | 86.09% |
| Highest Validation Accuracy | 87.56% |
| Test Loss | 0.3987 |
| Test Accuracy | 87.20% |

The final model was selected using the lowest validation loss because validation loss gives a better indication of model generalization than accuracy alone.

---

## Evaluation

The saved model was evaluated on the test set.

The evaluation notebook includes:

- Test loss
- Test accuracy
- Training loss vs validation loss plot
- Training accuracy vs validation accuracy plot
- Best validation loss plot
- Confusion matrix
- Classification report

The confusion matrix helped identify which classes the model confused the most. Some expected confusion occurred between visually similar classes such as glacier and mountain, and buildings and street.

---

## Project Structure
Expected local structure after downloading the dataset:

```text
intel-image-cnn/
│
├── data/
│   ├── seg_train/
│   └── seg_test/
│
├── notebooks/
├── plots/
├── best_model.pt
├── training_metrics.csv
├── requirements.txt
├── README.md
└── .gitignore
```

---
## Notes on Model File

The trained model weights are saved in:

```text
best_model.pt
```
---

## Conclusion

This project demonstrates a complete CNN image classification workflow using PyTorch. The model achieved a test accuracy of 87.20%, which is a decent result for an educational custom CNN project.

The project also shows the importance of:

- Data augmentation
- Validation tracking
- Early stopping
- Learning rate scheduling
- Saving the best model
- Confusion matrix analysis
