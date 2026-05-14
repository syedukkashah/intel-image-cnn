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

Download the dataset from:

```text
PASTE_DATASET_LINK_HERE
```

The dataset is not included in this repository because it is large.

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

### Training Set

| Class | Number of Images |
|---|---:|
| Buildings | 2191 |
| Forest | 2271 |
| Glacier | 2404 |
| Mountain | 2512 |
| Sea | 2274 |
| Street | 2382 |

### Test Set

| Class | Number of Images |
|---|---:|
| Buildings | 437 |
| Forest | 474 |
| Glacier | 553 |
| Mountain | 525 |
| Sea | 510 |
| Street | 501 |

The original training set was split into:

- 85% training data
- 15% validation data

---

## Image Preprocessing and Augmentation

Training images were resized to `150 x 150`.

The following transformations were applied to the training set:

```python
transforms.Resize((150, 150))
transforms.RandomHorizontalFlip(p=0.5)
transforms.RandomRotation(degrees=15)
transforms.ColorJitter(
    brightness=0.3,
    contrast=0.3,
    saturation=0.3
)
transforms.ToTensor()
transforms.Normalize([0.5]*3, [0.5]*3)
```

Validation and test images were not augmented. They were only resized, converted to tensors, and normalized.

This was done so that the training data could be made more varied, while validation and test results stayed consistent and reliable.

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

```text
intel-image-cnn/
│
├── notebooks/
│   ├── 01_training.ipynb
│   └── 02_evaluation_plots.ipynb
│
├── plots/
│   ├── loss_plot.png
│   ├── accuracy_plot.png
│   ├── best_val_loss_plot.png
│   ├── confusion_matrix.png
│   ├── classification_report.txt
│   └── classification_report.csv
│
├── best_model.pt
├── training_metrics.csv
├── requirements.txt
├── README.md
└── .gitignore
```

The `data/` folder is intentionally not included in this repository because of its size.

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

## How to Run the Project

### 1. Clone the repository

```bash
git clone PASTE_YOUR_REPOSITORY_LINK_HERE
cd intel-image-cnn
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the dataset

Download the dataset from:

```text
PASTE_DATASET_LINK_HERE
```

Place the dataset inside the `data/` folder using this structure:

```text
data/
├── seg_train/
└── seg_test/
```

### 4. Train the model

Open and run:

```text
notebooks/01_training.ipynb
```

This notebook trains the CNN and saves:

```text
best_model.pt
training_metrics.csv
```

### 5. Evaluate the model

Open and run:

```text
notebooks/02_evaluation_plots.ipynb
```

This notebook loads the saved model and generates:

- Test loss
- Test accuracy
- Metric plots
- Confusion matrix
- Classification report

---

## Requirements

Main libraries used:

```text
torch
torchvision
numpy
pandas
matplotlib
scikit-learn
Pillow
```

---

## Notes on Model File

The trained model weights are saved in:

```text
best_model.pt
```

For this educational project, the model file may be included if it is small enough. In larger production projects, model weights are usually stored separately using Git LFS, release assets, or external storage instead of being committed directly to Git.

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
