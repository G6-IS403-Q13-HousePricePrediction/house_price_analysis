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

# %% [markdown]
# ## Import the Packages

# %%
import numpy as np
import pandas as pd 
import os
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats

from math import sqrt
from sklearn.model_selection import KFold, cross_val_score

import optuna
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

import warnings
warnings.filterwarnings('ignore')

pd.pandas.set_option('display.max_columns', None)

# %% [markdown]
# ## Load the Datasets

# %%
df_train = pd.read_csv(os.path.join("..", "data", "train.csv"))
df_test = pd.read_csv(os.path.join("..", "data", "test.csv"))
sample_submission = pd.read_csv(os.path.join("..", "data", "sample_submission.csv"))
print("train data shape is {}".format(df_train.shape))
print("test data shape is {}".format(df_test.shape))
print("sample_submission data shape is {}".format(sample_submission.shape))

# %% [markdown]
# ## Check the Data

# %%
df_train.head()

# %%
# Check for duplicates in train set
df_train.duplicated().sum()

# %% [markdown]
# ## Target Feature Distribution
# Find out the distribution of the target feature

# %%
# Distribution of values in target feature
sns.distplot(df_train.get('SalePrice'), kde=False)
plt.show()

# %% [markdown]
# The dependent feature 'SalePrice' is right-skewed, we will have to perform a log-normal transformation on this feature.

# %% [markdown]
# ## Outliers

# %% [markdown]
# ### Find outliers for all the numerical dataset (before handling missing values)

# %%
numerical_df = df_train.select_dtypes(exclude=['object'])
numerical_df = numerical_df.drop(["Id"], axis=1)

for column in numerical_df:
    plt.figure(figsize=(16, 4))
    sns.set_theme(style='whitegrid')
    sns.boxplot(numerical_df[column], orient='h')
    plt.xlabel(column)
    plt.show()

# %% [markdown]
# ## Merge the datasets
# Combine the two datasets to do the pre-processing on before splitting it up again for the models

# %%
# Merge the dataframes together
df = pd.concat([df_train, df_test])

# Reset the index
df.reset_index(drop=True, inplace=True)

df.shape

# %%
df.head()

# %% [markdown]
# ## Fill in Missing Data
#
# Go through each feature and fill in the missing data.
# - Numerical: Fill in missing data with either 0 (which represents "None"), the mean or the mode, whichever makes more sense. Except for SalePrice.
# - Categorical: Fill in the data with "None" if it seems that it makes sense. Otherwise, fill it in with the mode.

# %%
# Find out missing rows
df.isnull().sum().sort_values(ascending=False).head(36)

# %%
print(df['PoolQC'].unique())
print("")
print(df['PoolQC'].value_counts())

# %%
df['PoolQC'].fillna('None', inplace=True)

# %%
print(df['MiscFeature'].unique())
print("")
print(df['MiscFeature'].value_counts())

# %%
df['MiscFeature'].fillna('None', inplace=True)

# %%
print(df['Alley'].unique())
print("")
print(df['Alley'].value_counts())

# %%
df['Alley'].fillna('None', inplace=True)

# %%
print(df['Fence'].unique())
print("")
print(df['Fence'].value_counts())

# %%
df['Fence'].fillna('None', inplace=True)

# %%
print(df['MasVnrType'].unique())
print("")
print(df['MasVnrType'].value_counts())

# %%
df['MasVnrType'].fillna('None', inplace=True)

# %%
print(df['FireplaceQu'].unique())
print("")
print(df['FireplaceQu'].value_counts())

# %%
df['FireplaceQu'].fillna('None', inplace=True)

# %%
print(df['LotFrontage'].unique())
print("")
print(df['LotFrontage'].value_counts())

# %%
df['LotFrontage'].fillna(0, inplace=True)

# %%
print(df['GarageYrBlt'].unique())
print("")
print(df['GarageYrBlt'].value_counts())

# %%
df['GarageYrBlt'].fillna(0, inplace=True)

# %%
# Fix error of 2207 year entry to 2007
df['GarageYrBlt'] = df['GarageYrBlt'].replace(2207, 2007)

# %%
print(df['GarageFinish'].unique())
print("")
print(df['GarageFinish'].value_counts())

# %%
df['GarageFinish'].fillna("None", inplace=True)

# %%
print(df['GarageQual'].unique())
print("")
print(df['GarageQual'].value_counts())

# %%
df['GarageQual'].fillna("None", inplace=True)

# %%
print(df['GarageCond'].unique())
print("")
print(df['GarageCond'].value_counts())

# %%
df['GarageCond'].fillna("None", inplace=True)

# %%
print(df['GarageType'].unique())
print("")
print(df['GarageType'].value_counts())

# %%
df['GarageType'].fillna("None", inplace=True)

# %%
print(df['BsmtExposure'].unique())
print("")
print(df['BsmtExposure'].value_counts())

# %%
df['BsmtExposure'].fillna("None", inplace=True)

# %%
print(df['BsmtCond'].unique())
print("")
print(df['BsmtCond'].value_counts())

# %%
df['BsmtCond'].fillna("None", inplace=True)

# %%
print(df['BsmtQual'].unique())
print("")
print(df['BsmtQual'].value_counts())

# %%
df['BsmtQual'].fillna("None", inplace=True)

# %%
print(df['BsmtFinType2'].unique())
print("")
print(df['BsmtFinType2'].value_counts())

# %%
df['BsmtFinType2'].fillna("None", inplace=True)

# %%
print(df['BsmtFinType1'].unique())
print("")
print(df['BsmtFinType1'].value_counts())

# %%
df['BsmtFinType1'].fillna("None", inplace=True)

# %%
# print(df['MasVnrArea'].unique())
print("")
print(df['MasVnrArea'].value_counts())

# %%
df['MasVnrArea'].fillna(0, inplace=True)

# %%
print(df['MSZoning'].unique())
print("")
print(df['MSZoning'].value_counts())

# %%
df['MSZoning'].fillna('RL', inplace=True)

# %%
print(df['Functional'].unique())
print("")
print(df['Functional'].value_counts())

# %%
df['Functional'].fillna('Typ', inplace=True)

# %%
print(df['BsmtFullBath'].unique())
print("")
print(df['BsmtFullBath'].value_counts())

# %%
df['BsmtFullBath'].fillna(0, inplace=True)

# %%
print(df['Utilities'].unique())
print("")
print(df['Utilities'].value_counts())

# %%
df['Utilities'].fillna('AllPub', inplace=True)

# %%
print(df['BsmtHalfBath'].unique())
print("")
print(df['BsmtHalfBath'].value_counts())

# %%
df['BsmtHalfBath'].fillna(0, inplace=True)

# %%
print(df['Electrical'].unique())
print("")
print(df['Electrical'].value_counts())

# %%
df['Electrical'].fillna('SBrkr', inplace=True)

# %%
print(df['TotalBsmtSF'].unique())
print("")
print(df['TotalBsmtSF'].value_counts())

# %%
df['TotalBsmtSF'].fillna(0, inplace=True)

# %%
print(df['BsmtUnfSF'].unique())
print("")
print(df['BsmtUnfSF'].value_counts())

# %%
df['BsmtUnfSF'].fillna(0, inplace=True)

# %%
print(df['KitchenQual'].unique())
print("")
print(df['KitchenQual'].value_counts())

# %%
# df['KitchenQual'].fillna("None", inplace=True)
df['KitchenQual'].fillna("TA", inplace=True)

# %%
#print(df['BsmtFinSF2'].unique())
print("")
print(df['BsmtFinSF2'].value_counts())

# %%
df['BsmtFinSF2'].fillna(0, inplace=True)

# %%
#print(df['BsmtFinSF1'].unique())
print("")
print(df['BsmtFinSF1'].value_counts())

# %%
df['BsmtFinSF1'].fillna(0, inplace=True)

# %%
print(df['SaleType'].unique())
print("")
print(df['SaleType'].value_counts())

# %%
df['SaleType'].fillna('WD', inplace=True)

# %%
print(df['GarageCars'].unique())
print("")
print(df['GarageCars'].value_counts())

# %%
df['GarageCars'].fillna(2, inplace=True)
# df['GarageCars'].fillna(0, inplace=True)

# %%
print(df['Exterior2nd'].unique())
print("")
print(df['Exterior2nd'].value_counts())

# %%
df['Exterior2nd'].fillna('VinylSd', inplace=True)

# %%
print(df['Exterior1st'].unique())
print("")
print(df['Exterior1st'].value_counts())

# %%
df['Exterior1st'].fillna('VinylSd', inplace=True)

# %%
# print(df['GarageArea'].unique())
print("")
print(df['GarageArea'].value_counts())

# %%
df['GarageArea'].fillna(0, inplace=True)

# %%
# Find out missing rows
df.isnull().sum().sort_values(ascending=False).head()

# %% [markdown]
# ## Changing Data Types

# %%
# Change floats to int to make it better
df['BsmtFullBath'] = df['BsmtFullBath'].astype(int)
df['BsmtHalfBath'] = df['BsmtHalfBath'].astype(int)
df['GarageYrBlt'] = df['GarageYrBlt'].astype(int)
df['GarageCars'] = df['GarageCars'].astype(int)

# %% [markdown]
# ## Feature Engineering

# %%
# Look up correlation between numeric features

# Select only numeric columns for calculating the correlation matrix
numeric_df = df.select_dtypes(include='number')

# Calculate the correlation matrix
df_corr = numeric_df.corr().round(2)

# Create a heatmap
plt.figure(figsize=(48, 32))
sns.heatmap(df_corr, annot=True, cmap='coolwarm', fmt="0.2f", linewidths=0.5, annot_kws={"size": 16})

plt.show()

# %%
# Features that have years in them
df['GarageYrBlt'] = df['YrSold'] - df['GarageYrBlt']
df['YearBuilt'] = df['YrSold'] - df['YearBuilt']
df['YearRemodAdd'] = df['YrSold'] - df['YearRemodAdd']

df.drop(['YrSold'], axis=1, inplace=True)
df.drop(['MoSold'], axis=1, inplace=True) #Isn't necessary

# %%
# Features that have square feet
df['BsmtFinSF'] = df['BsmtFinSF1'] + df['BsmtFinSF2']
df['TotalFlrSF'] = df['1stFlrSF'] + df['2ndFlrSF']

df.drop(['BsmtFinSF1'], axis=1, inplace=True)
df.drop(['BsmtFinSF2'], axis=1, inplace=True)
df.drop(['1stFlrSF'], axis=1, inplace=True)
df.drop(['2ndFlrSF'], axis=1, inplace=True)
df.drop(['TotalBsmtSF'], axis=1, inplace=True) # Isn't necessary since it's a sum of the other columns

# %%
# Features that are about bathrooms
df['TotalBaths'] = df['FullBath'] + (0.5*df['HalfBath']) + df['BsmtFullBath'] + (0.5*df['BsmtHalfBath'])

df.drop(['FullBath'], axis=1, inplace=True)
df.drop(['HalfBath'], axis=1, inplace=True)
df.drop(['BsmtFullBath'], axis=1, inplace=True)
df.drop(['BsmtHalfBath'], axis=1, inplace=True)

# %%
# Features about the garage
df['GarageAreaPerCar'] = df['GarageArea'] / df['GarageCars']

# Fill in any nulls from feature engineering
df['GarageAreaPerCar'].fillna(0, inplace=True)

df.drop(['GarageArea'], axis=1, inplace=True)
df.drop(['GarageCars'], axis=1, inplace=True)

# %% [markdown]
# ## Feature Transformation
#
# Feature transformation refers to the process of altering the features or variables in the dataset to make them more suitable for analysis or modeling. This transformation can involve various techniques to modify the distribution, scale, or relationships between the features. Feature transformation is a crucial step in data preprocessing, particularly in machine learning tasks, where the quality and characteristics of the input features significantly impact the performance of the models.

# %%
# Create a list of the numerical data types after the feature engineering
numerical_cols = [cname for cname in df.columns if df[cname].dtypes!='object' and cname!='SalePrice']

# Create new dataframe with the numerical columns
skew_df = pd.DataFrame(numerical_cols, columns=['Feature'])

# This function used to compute the skewness of a dataset
skew_df['Skew'] = skew_df['Feature'].apply(lambda feature: scipy.stats.skew(df[feature]))

# Change Skew to a postive number
skew_df['Absolute Skew'] = skew_df['Skew'].apply(abs)

# Create true/false columns based on if Absolute Skew is >=0.5
skew_df['Skewed'] = skew_df['Absolute Skew'].apply(lambda x: True if x >= 0.5 else False)

# %%
skew_df

# %%
# Check if a column has a min of 0 for log transformation
df[numerical_cols].describe()

# %%
# Apply log1p transformation to df whose above a 0.5 absolute skew
for column in skew_df.query("Skewed == True")['Feature'].values:
    df[column] = np.log1p(df[column])

# %% [markdown]
# ### Encoding Categorical

# %%
# Get list of numerical before starting
num_cols = df.select_dtypes(include=['int64', 'float64']).columns
num_cols = num_cols.drop('SalePrice')
num_cols = num_cols.drop('Id')

# %%
num_cols

# %%
# Get list of categorical
categorical_columns_list = df.select_dtypes(include='object').columns

# Create a new Dataframe containing only the categorical columns
df_categorical = df[categorical_columns_list].copy()

df_categorical

# %% [markdown]
# The label encoder method was giving undesirable values, for example "None" could of been a 4, so going to do it manually
#
#

# %%
# Label encode all the ordinal data
ordinal_1 = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC', 'KitchenQual', 'FireplaceQu', 
                 'GarageQual', 'GarageCond', 'PoolQC']

for col in ordinal_1:
    if 'None' in df[col].value_counts().index:
        df[col] = df[col].map({"None":0,"Po":1,"Fa":2,"TA":3,"Gd":4,"Ex":5}).astype('int')
    else:
        df[col] = df[col].map({"Po":1,"Fa":2,"TA":3,"Gd":4,"Ex":5}).astype('int')

# %%
# Label encode all the ordinal data individually, and add to list

ordinal_2 = ['LotShape', 'LandContour', 'LandSlope', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
                 'Utilities', 'CentralAir', 'Functional', 'GarageFinish', 'PavedDrive', 'Fence']

df['LotShape'] = df['LotShape'].map({"None":0,"IR3":1,"IR2":2,"IR1":3,"Reg":4}).astype('int')
df['LandContour'] = df['LandContour'].map({"None":0,"Low":1,"Bnk":2,"HLS":3,"Lvl":4}).astype('int')
df['LandSlope'] = df['LandSlope'].map({"None":0,"Sev":1,"Mod":2,"Gtl":3}).astype('int')
df['BsmtExposure'] = df['BsmtExposure'].map({"None":0, "No":1,"Mn":2,"Av":3,"Gd":4}).astype('int')
df['BsmtFinType1'] = df['BsmtFinType1'].map({"None":0, "Unf":1,"LwQ":2,"Rec":3,"BLQ":4,"ALQ":5,"GLQ":6}).astype('int')
df['BsmtFinType2'] = df['BsmtFinType2'].map({"None":0, "Unf":1,"LwQ":2,"Rec":3,"BLQ":4,"ALQ":5,"GLQ":6}).astype('int')
df['Utilities'] = df['Utilities'].map({"None":0,"ELO":1,"NoSeWa":2,"NoSewr":3,"AllPub":4}).astype('int')
df['CentralAir'] = df['CentralAir'].map({"None":0,"N":1,"Y":2}).astype('int')
df['Functional'] = df['Functional'].map({"None":0,"Sal":1,"Sev":2,"Maj2":3,"Maj1":4,"Mod":5,"Min2":6,
                                         "Min1":7,"Typ":8}).astype('int')
df['GarageFinish'] = df['GarageFinish'].map({"None":0,"Unf":1,"RFn":2,"Fin":3}).astype('int')
df['PavedDrive'] = df['PavedDrive'].map({"None":0,"N":1,"P":2,"Y":3}).astype('int')
df['Fence'] = df['Fence'].map({"None":0, "MnWw":1,"GdWo":2,"MnPrv":3,"GdPrv":4}).astype('int')

# %%
# Get remaining non-ordinal categorical columns
df_categorical = df_categorical.drop(columns=ordinal_1)
df_categorical = df_categorical.drop(columns=ordinal_2)

# Get the rest of the categorical data for one hot encoding
ohe_ = df_categorical.columns.tolist()

# Convert categorical columns to one-hot encoded columns, dropping the first column for each
df_encoded = pd.get_dummies(df[ohe_], drop_first=True).astype(int)

# Concatenate the one-hot encoded columns with the original Dataframe
df = pd.concat([df.drop(columns=ohe_), df_encoded], axis=1)

# %% [markdown]
# ## Split Train and Test Data

# %%
# Split the data back into train and test sets
df_train = df.iloc[:df_train.shape[0]]
df_test = df.iloc[df_train.shape[0]:]

df_train.drop(['Id'], axis=1, inplace=True)
df_test.drop(['SalePrice', 'Id'], axis=1, inplace=True)

# %% [markdown]
# ## Target Feature Encoding
#
# Perform log transformation on SalePrice so it's less skewed

# %%
log_target = np.log(df_train['SalePrice'])

df_train.drop(['SalePrice'], axis=1, inplace=True)

# %% [markdown]
# ## Optuna Parameters

# %%
# # Define objective function for Optuna for catboost
# def objective_cat(trial):
#     # Define hyperparameters to optimize
#     catboost_params = {
#         'iterations': trial.suggest_int('iterations', 1000, 8000),
#         'learning_rate': trial.suggest_loguniform('learning_rate', 0.005, 0.08),
#         'depth': trial.suggest_int('depth', 3, 7),
#         'eval_metric': 'RMSE',
#     }

#     # Initialize models with suggested parameters
#     catboost_model = CatBoostRegressor(**catboost_params, verbose=0)
    
#     # Train models
#     catboost_model.fit(df_train, log_target)

#     # Calculate RMSE
#     kf = KFold(n_splits=10)
#     catboost_rmse = np.exp(np.sqrt(-cross_val_score(catboost_model, df_train, log_target, scoring='neg_mean_squared_error', cv=kf)))

#     # Return average RMSE
#     return np.mean(catboost_rmse)

# %%
# # Define objective function for Optuna for xgboost
# def objective_xgb(trial):
#     # Define hyperparameters to optimize
#     xgboost_params = {
#         'n_estimators': trial.suggest_int('n_estimators', 1000, 8000),
#         'learning_rate': trial.suggest_loguniform('learning_rate', 0.005, 0.08),
#         'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.2, 0.6),
#         'subsample': trial.suggest_uniform('subsample', 0.4, 0.8),
#         'min_child_weight': trial.suggest_int('min_child_weight', 2, 5),
#     }

#     # Initialize models with suggested parameters
#     xgb_model = XGBRegressor(**xgboost_params, verbosity=0)

#     # Train models
#     xgb_model.fit(df_train, log_target)

#     # Calculate RMSE
#     kf = KFold(n_splits=10)
#     xgb_rmse = np.exp(np.sqrt(-cross_val_score(xgb_model, df_train, log_target, scoring='neg_mean_squared_error', cv=kf)))

#     # Return average RMSE
#     return np.mean(xgb_rmse)

# %%
# # Optimize hyperparameters catboost
# study_cat = optuna.create_study(direction='minimize')
# study_cat.optimize(objective_cat, n_trials=50)

# %%
# # Optimize hyperparameters xgboost
# study_xgb = optuna.create_study(direction='minimize')
# study_xgb.optimize(objective_xgb, n_trials=50)

# %%
# # Get best parameters
# best_params_cat = study_cat.best_params
# best_params_xgb = study_xgb.best_params

# %%
# best_params_cat

# %%
# study_cat.best_value

# %%
# best_params_xgb

# %%
# study_xgb.best_value

# %% [markdown]
# ## Forming and Testing the Model

# %%
best_params_cat = {
    'iterations': 6623,
    'learning_rate': 0.01711,
    'depth': 5,
    'eval_metric':'RMSE',
}

best_params_xgb = {
    'n_estimators': 6696,
    'learning_rate': 0.00630,
    'colsample_bytree': 0.22301,
    'subsample': 0.45878,
    'min_child_weight': 3,
}

# %%
# Use best parameters to train final models
catboost_model = CatBoostRegressor(**best_params_cat, verbose=0)
xgb_model = XGBRegressor(**best_params_xgb, verbosity=0)

catboost_model.fit(df_train, log_target)
xgb_model.fit(df_train, log_target)

# %%
# Calculate RMSE CatBoost
kf = KFold(n_splits=10)
catboost_rmse = np.sqrt(-cross_val_score(catboost_model, df_train, log_target, scoring='neg_mean_squared_error', cv=kf))

# %%
xgb_rmse = np.sqrt(-cross_val_score(xgb_model, df_train, log_target, scoring='neg_mean_squared_error', cv=kf))

# %%
# Return average RMSE
print(np.mean(catboost_rmse))
print(np.mean(xgb_rmse))

# %%
# Combine predictions
final_predictions = (
    0.80 * np.exp(catboost_model.predict(df_test)) + 
    0.20 * np.exp(xgb_model.predict(df_test))
)

# %%
output = pd.DataFrame({'Id': df_test.index+1, 'SalePrice': final_predictions})

# %%
output

# %%
output.to_csv('submission.csv', index=False)

# %%
