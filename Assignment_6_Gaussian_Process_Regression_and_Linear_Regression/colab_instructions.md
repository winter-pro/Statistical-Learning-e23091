# Colab Instructions

Follow these steps to run the assignment in Google Colab.

## Step 1: Create a New Colab Notebook

Go to Google Colab and create a new notebook.

## Step 2: Install Requirements

```python
!pip install -r requirements.txt
```

If you are running directly without cloning from GitHub, install the libraries manually:

```python
!pip install kagglehub pandas numpy matplotlib scikit-learn
```

## Step 3: Run Gaussian Process Regression

```python
!python src/gp_energy_efficiency.py
```

This script will:

- Download the Energy Efficiency dataset.
- Check the relationship between heating load and cooling load.
- Create one latent parameter using PCA.
- Train a single Gaussian Process.
- Compare it with separate Gaussian Process models.
- Save plots in the `figures/` folder.

## Step 4: Run Linear Regression

```python
!python src/linear_green_building.py
```

This script will:

- Download the Green Building dataset.
- Select useful predictors.
- Train a Linear Regression model.
- Print RMSE, MAE, and R².
- Save actual-vs-predicted and residual plots in the `figures/` folder.

## Step 5: Submit

For submission, include:

- Source code files from `src/`
- `README.md`
- `reports/discussion.md`
- Generated plots from `figures/`
- Your Colab notebook link, if required
