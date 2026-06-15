# Assignment Discussion

## 1. Gaussian Process Regression

The Energy Efficiency dataset contains eight input features, denoted by `X1` to `X8`, and two output responses, denoted by `Y1` and `Y2`.

In this dataset:

- `Y1` represents heating load.
- `Y2` represents cooling load.

The aim was to explore whether heating load and cooling load can be modeled as a single-parameter Gaussian Process.

### Method

First, the correlation between heating load and cooling load was examined. If the two outputs are highly correlated, it suggests that both responses are controlled by a common underlying thermal behavior of the building.

Next, both outputs were standardized and Principal Component Analysis was applied. PCA was used to convert the two responses into a single latent variable. This latent variable can be interpreted as a common thermal-load parameter.

A Gaussian Process Regression model was then trained using the eight input features and the single latent output parameter. After prediction, the latent parameter was transformed back into heating load and cooling load.

For comparison, two separate Gaussian Process models were also trained:

- One model for heating load.
- One model for cooling load.

### Conclusion

Heating load and cooling load are strongly related, so it is possible to approximately represent them using one latent thermal-load parameter. Therefore, a single Gaussian Process model can be used for compact modeling.

However, this method is an approximation. The single latent parameter may not preserve all differences between heating and cooling behavior. Therefore, if the main goal is maximum prediction accuracy, separate Gaussian Process models or a proper multi-output Gaussian Process model are more suitable.

The single-parameter GP is useful when a simpler model and easier interpretation are preferred.

## 2. Linear Regression

The Green Building Multi-Source Environment dataset contains building and environmental data. The aim was to predict `predicted_energy_demand` using a linear relationship with a suitable set of other parameters.

### Method

The target variable `predicted_energy_demand` was removed from the input set. ID columns and timestamp/date columns were also removed because they usually do not represent meaningful physical causes of energy demand.

The remaining numerical and categorical variables were preprocessed:

- Missing numerical values were replaced using the median.
- Numerical variables were standardized.
- Missing categorical values were replaced using the most frequent value.
- Categorical variables were one-hot encoded.

Lasso regression was used for feature selection. Lasso is suitable because it can reduce the coefficients of weak predictors to zero. The selected predictors were then used in a final Linear Regression model.

The model was evaluated using:

- RMSE
- MAE
- R²

### Conclusion

Linear Regression is useful as a simple and interpretable baseline model for predicting energy demand. The selected parameters are justified because they show measurable contribution to the target variable through the coefficient analysis.

If the R² value is high and the residual plot is randomly scattered around zero, the linear model is suitable. If the residual plot shows a pattern or curved shape, it means the relationship is not fully linear and nonlinear models should be considered.

Overall, the Linear Regression model provides a clear starting point for energy demand prediction, while more advanced models can be used if higher accuracy is required.
