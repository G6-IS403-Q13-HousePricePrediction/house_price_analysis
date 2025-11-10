# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# %%
# --- 1. Create Sample Data ---
X_data = {
    'Size_sqft': [1500, 2400, 1800, 1900, 3000, 2200],
    'Bedrooms': [3, 4, 3, 3, 5, 4],
    'Age_years': [10, 2, 8, 20, 5, 12]
}
y_data = [300000, 550000, 390000, 340000, 680000, 480000]

X = pd.DataFrame(X_data)
y = np.array(y_data)

# %%
# --- 2. Split the Data ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %%
# --- 3. Define the Model ---
rf = RandomForestRegressor(random_state=42)

# %%
# --- 4. Define the "Grid" of Hyperparameters ---
param_grid = {
    'n_estimators': [50, 100, 200],  # Try 50, 100, or 200 trees
    'max_depth': [None, 10, 30],       # Try no limit, 10, or 30 for depth
    'min_samples_leaf': [1, 2, 4]      # Try 1, 2, or 4 samples per leaf
}
# Total combinations to test: 3 * 3 * 3 = 27

# %%
# --- 5. Set Up GridSearchCV ---
# estimator: the model we want to tune
# param_grid: the grid of settings we just defined
# cv: the number of cross-validation folds (e.g., 5)
# scoring: the metric to judge the "best" model (e.g., for regression, use 'neg_mean_squared_error')
# n_jobs: -1 uses all your computer's cores to speed it up
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=2,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    verbose=2  # This will print updates so you can see its progress
)

# %%
# --- 6. Run the Search ---
# This is the step that takes a long time!
# It's training 27 combinations * 5 folds = 135 models.
grid_search.fit(X_train, y_train)

# %%
# --- 7. Get the Best Results ---
# Print the best combination of settings it found
print(f"Best hyperparameters found: {grid_search.best_params_}")

# Get the best-performing model, which is already re-trained on all our data
best_model = grid_search.best_estimator_

# We can now use this 'best_model' to make predictions
y_pred = best_model.predict(X_test)

# %%
# 8. Calculate the RMSE
# First, get the MSE
mse = mean_squared_error(y_test, y_pred)

# Then, get the RMSE
test_rmse = np.sqrt(mse)

print(f"The final RMSE on the test set is: {test_rmse}")

# %%

# %%

# %%
