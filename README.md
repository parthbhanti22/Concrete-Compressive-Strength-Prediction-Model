# Concrete Compressive Strength Prediction using Neural Networks

This repository contains four Jupyter Notebook implementations for predicting the compressive strength of concrete using feedforward neural networks (FNN) built with Keras and TensorFlow. Each notebook explores a different aspect of model development and optimization.

---

## Project Overview

The objective is to model the compressive strength of concrete based on features such as the amount of cement, water, and other aggregates. The project involves the following steps:

1. Building a baseline model.
2. Normalizing the data and retraining the model.
3. Increasing the number of epochs to evaluate training duration impact.
4. Increasing the number of hidden layers to enhance model complexity.

---

## Contents

### 1. **Part A: Baseline Model**

* **File:** `baseline_model.ipynb`
* **Description:** Implements a simple feedforward neural network with one hidden layer (10 nodes, ReLU activation). Evaluates performance across 50 iterations and computes the mean and standard deviation of mean squared errors (MSEs).
* **Key Features:**
  * Adam optimizer
  * Mean squared error as the loss function
  * 50 epochs for training

### 2. **Part B: Normalized Data**

* **File:** `normalized_data.ipynb`
* **Description:** Normalizes input features and retrains the baseline model. The effect of normalization on model performance is analyzed by comparing the mean and standard deviation of MSEs.
* **Normalization Method:**
  * Subtracted the mean and divided by the standard deviation of each predictor.

### 3. **Part C: Increased Epochs**

* **File:** `increased_epochs.ipynb`
* **Description:** Extends the training duration to 100 epochs while keeping the data normalized. The impact of increased epochs is evaluated by comparing the results with the baseline (50 epochs).

### 4. **Part D: Increased Hidden Layers**

* **File:** `increased_hidden_layers.ipynb`
* **Description:** Adds three hidden layers (each with 10 nodes, ReLU activation) to the neural network while using normalized data and training for 50 epochs. Performance is compared to the single hidden-layer model.

---

## Data Description

The dataset is sourced from the [Concrete Compressive Strength Dataset](https://cocl.us/concrete_data), with the following features:

* **Predictors:** Cement, Blast Furnace Slag, Fly Ash, Water, Superplasticizer, Coarse Aggregate, Fine Aggregate, and Age.
* **Target Variable:** Compressive Strength of concrete.

---

## Usage

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/concrete-strength-regression.git
    ```

2. Open the Jupyter Notebooks in your environment or use Google Colab.
3. Ensure the required dependencies are installed:
    * TensorFlow
    * Scikit-learn
    * Pandas
    * NumPy
4. Run the notebooks sequentially to observe results for each part.

---

## Results Summary

| **Part**                        | **Mean MSE**          | **Std Dev of MSE**     |
|----------------------------------|-----------------------|------------------------|
| Baseline Model (Part A)          | 116.68                | 27.78                  |
| Normalized Data (Part B)         | 130.48                | 11.15                  |
| 100 Epochs (Part C)              | 78.85                 | 13.23                  |
| 3 Hidden Layers (Part D)         | 82.90                 | 20.69                  |

---

## License

This project is open-source and available under the MIT License.
