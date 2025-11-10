- A **coefficient** is a number that tells us the importance and direction of
  a feature's relationship with our target variable.
- **Overfitting** is when our model learns the training data too well. It becomes
  too complex and starts to memorize the noise in our data, not just the true,
  underlying **pattern**.
- **Multicollinearity** is when two or more of our features are highly correlated
  with each other. This is a problem because it "confuses" the model, which
  doesn't know which feature to give "credit" to.
- **n_estimators**: How many "specialist" trees to build.
- **learning_rate**: How much each tree's correction is applied (a low value
  prevents overfitting).
- **max_depth**: How complex each individual tree can be (a low depth prevents
  overfitting).
- **Regularization** changes the model's goal. It says, "You still need to
  minimize error, but you will now be penalized for being too complex."
- **Feature engineering:** data preprocessing
- **RMSE** (Root Mean Square Error): average error of model's predictions, in the
  same units as the target. **Lower is better**.
- **R-squared** (R²): Tells how much of the target's variation model can explain.
  It's a percentage. **Higher is better**.
- A **scatter plot** is a type of chart used to visualize the relationship between
  two numerical variables. The main goal of a scatter plot is to see if a
  pattern or correlation exists between the two variables.
- A **box plot** is a chart that shows the distribution and spread of numerical
  data (SalePrice) grouped by categories (Alley).
- A **residual** is the error for a single prediction. (Res = AC - PV)
- **One-hot encoding** is a process used to convert **categorical data** into a
  numerical format that algorithms can understand. It takes a column with
  categorical values (like "Red", "Green", "Blue") and transforms it into
  multiple new columns—one for each unique category. A `1` is placed in the
  column corresponding to the original value, and `0`s are placed in all other
  new columns for that row.
- **Ordinal encoding** is a method for converting categorical data into integers,
  but it's used specifically when the categories have a meaningful, natural
  order (an "ordinal" relationship).
- **SimpleImputer**: find all the missing values (like NaN, or "Not a Number") in
  the dataset and fill them in using a specific rule, or "strategy."
- **StandardScaler** is a data preprocessing technique that **rescales numerical data
  to have a mean of 0 and a standard deviation of 1**.
- A **fold** is just one of several equal-sized chunks (or subsets) of the
  training data.
- The **RandomForestRegressor** is an ensemble machine learning model that predicts
  a continuous value (like a price or temperature) by averaging the predictions
  of many individual decision trees. When we ask the **forest** to make a new
  prediction, it asks all 100 "experts." For regression, it averages their
  answers. The individual errors and biases cancel each other out, resulting in
  a single, stable, and highly accurate prediction.
- **GridSearchCV** is a tool that automates the process of hyperparameter tuning.
  It systematically searches for the best combination of model settings by
  trying every single one we specify. (CV = Cross Validation)
- **XGBRegressor (eXtreme Gradient Boosting)** is one of the most powerful and
  popular machine learning models for regression. It's an implementation of
  **Gradient Boosting,** which, like **Random Forest**, is an ensemble model
  that combines many decision trees.
- **RandomForestRegressor** vs **XGBRegressor**:
  - **Random Forest** is a "parallel" team. It builds 100 independent trees and
    then averages their predictions.
  - **XGBoost** is a "sequential" team. It builds one tree, which makes a
    prediction. A second tree is then built to correct the errors of the first
    tree. A third tree is built to correct the errors of the second, and so on.
- The **learning_rate** (also called eta) controls how much each new tree corrects
  the previous trees' errors.
- **Ridge** is a linear regression model that uses **regularization** to prevent
  overfitting. It's an improved version of LinearRegression that is especially
  good at handling data where our features are highly correlated (a problem
  called **multicollinearity**). The main idea is that it adds a **penalty** for
  large coefficients. This forces the model to find a "simpler" solution that
  generalizes better to new data.
- **GradientBoosting (Standard Gradient Boosting)** builds trees
  **sequentially, in a chain, where each new tree's job is to correct the
  errors of the tree that came before it**. "Gradient Boosting" means: each
  step moves the model in the direction (the "gradient") that minimizes the
  overall error.
- **LightGBM** (Light Gradient Boosting Machine): Builds a tree by always splitting
  the one leaf that gives the most improvement. It's a "best-first" search.
  This allows the model to converge much more quickly and find complex
  patterns.
- The purpose of **VotingRegressor** is to combine several different regression
  models into one single, more powerful "super-model."

## Models

Metric: Submissions are evaluated on Root-Mean-Squared-Error (RMSE) between the
logarithm of the predicted value and the logarithm of the observed sales price.
(Taking logs means that errors in predicting expensive houses and cheap houses
will affect the result equally.)

1. LinearRegression
    RMSE: 0.13312882636583184
    Score: 0.14378
2. RandomForestRegressor
3. XGBRegressor
4. Ridge
5. GradientBoostingRegressor
6. LGBMRegressor
7. Catboost
8. VotingRegressor
9. StackRegressor

<!-- 02:44:00 -->
