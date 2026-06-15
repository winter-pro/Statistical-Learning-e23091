# Energy Analysis Assignment

This repository contains the submission files for the assignment on:

1. Gaussian Process Regression for heating load and cooling load prediction.
2. Linear Regression for predicted energy demand in green building data.

The work uses two Kaggle datasets:

- Energy Efficiency Dataset  
  `elikplim/eergy-efficiency-dataset`

- Green Building Multi-Source Environment Dataset  
  `programmer3/green-building-multi-source-environment-dataset`

## Repository Structure

```text
energy_gp_linear_assignment/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── src/
│   ├── gp_energy_efficiency.py
│   └── linear_green_building.py
│
├── notebooks/
│   └── colab_instructions.md
│
├── reports/
│   └── discussion.md
│
└── figures/
    └── generated plots can be saved here
```

## Part 1: Gaussian Process Regression

The first part explores whether heating load and cooling load can be modeled using a single-parameter Gaussian Process.

The method used is:

1. Load the Energy Efficiency dataset.
2. Use features `X1` to `X8`.
3. Use outputs `Y1` and `Y2`, where:
   - `Y1` = Heating Load
   - `Y2` = Cooling Load
4. Check the correlation between `Y1` and `Y2`.
5. Use PCA to combine `Y1` and `Y2` into one latent thermal-load parameter.
6. Train a Gaussian Process model on this single latent target.
7. Compare the result with two separate Gaussian Process models.

## Part 2: Linear Regression

The second part explores whether `predicted_energy_demand` can be predicted using a linear relationship with suitable parameters.

The method used is:

1. Load the Green Building dataset.
2. Remove the target variable from the input features.
3. Remove non-informative ID or timestamp columns.
4. Preprocess numerical and categorical variables.
5. Use Lasso regression for feature selection.
6. Train a final Linear Regression model.
7. Evaluate using RMSE, MAE, and R².
8. Analyze the residual plot.

## How to Run in Google Colab

Open a new Google Colab notebook and run:

```python
!git clone https://github.com/YOUR_USERNAME/energy_gp_linear_assignment.git
%cd energy_gp_linear_assignment
!pip install -r requirements.txt
```

Then run:

```python
!python src/gp_energy_efficiency.py
!python src/linear_green_building.py
```

Replace `YOUR_USERNAME` with your actual GitHub username after uploading the repository.

## Main Conclusion

Heating load and cooling load are strongly related, so a single latent thermal-load parameter can approximately represent both outputs. A single Gaussian Process is useful for compact modeling, but separate Gaussian Process models or a multi-output Gaussian Process are more suitable when maximum accuracy is required.

For the second dataset, Linear Regression provides a simple and interpretable baseline for predicting energy demand. If the R² value is high and residuals are randomly scattered, the selected linear relationship is suitable. If residuals show a pattern, nonlinear models should be considered.
