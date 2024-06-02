
# Industrial Copper Modeling

This project aims to develop two machine learning models for the copper industry to address the challenges of predicting selling prices and lead classification. Manual predictions can be time-consuming and may not result in optimal pricing decisions or accurately capture leads.

The models will utilize advanced techniques such as data normalization, outlier detection and handling, handling data in the wrong format, identifying the distribution of features, and leveraging tree-based models, specifically the decision tree algorithm, to predict the selling price and leads accurately.



## Project Overview

### Project Title

Industrial Copper Modeling

### Skills Takeaway

Python scripting
Data Preprocessing
Exploratory Data Analysis (EDA)
Streamlit
Machine Learning (Regression and Classification)
Model Deployment

### Domain

Manufacturing

## Problem Statement

The copper industry faces challenges in sales and pricing due to less complex data, which may contain skewness and noisy data. Manual predictions may be inaccurate and time-consuming. This project aims to develop machine learning models to predict selling prices and lead statuses, addressing challenges such as data normalization, outlier detection, and leveraging regression and classification algorithms.

## Regression Model Details

The copper industry deals with less complex data related to sales and pricing. However, this data may suffer from issues such as skewness and noisy data, which can affect the accuracy of manual predictions. A machine learning regression model can address these issues by utilizing advanced techniques such as data normalization, outlier detection and handling, handling data in the wrong format, identifying the distribution of features, and leveraging tree-based models, specifically the decision tree algorithm.

## Classification Model Details

Another area where the copper industry faces challenges is in capturing leads. A lead classification model evaluates and classifies leads based on how likely they are to become a customer. The STATUS variable can be used, with WON considered as Success and LOST as Failure. Data points other than WON and LOST STATUS values should be removed.

## Solution

### Exploratory Data Analysis (EDA)

Identify variable types and distributions.

Treat 'Material_Reference' rubbish values starting with '00000'.

Treat reference columns as categorical variables.

Remove unnecessary 'INDEX'.

### Data Preprocessing

Handle missing values using mean/median/mode.

Treat outliers using IQR.

Identify and treat skewness using appropriate transformations.

Encode categorical variables using suitable techniques.

### Feature Engineering

Engineer new features if applicable.

Drop highly correlated columns using a heatmap.

### Model Building and Evaluation

Split the dataset into training and testing sets.

Train and evaluate regression models for 'Selling_Price'.

Train and evaluate classification models for 'Status' (WON/LOST.

Optimize model hyperparameters using cross-validation and grid search.

Interpret model results and assess performance.

### Model GUI (Streamlit)

Create an interactive page.

Input task (Regression/Classification) and column values.

Perform feature engineering, scaling, and transformation steps.

Predict new data and display the output.

### Extras:

Use the pickle module to dump and load models.

Fit and transform in separate lines, use transform only for unseen data.

## Lessons Learned:

Python and Data analysis libraries (Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, Streamlit).

Data Preprocessing.

Data visualizationthrough EDA techniques.

Application of regression and classification using Machine Learning.

Building and optimizing Machine Learning models.

Feature engineering skills.

Web application development using Streamlit.

Understanding challenges and best practices in the manufacturing domain.

## Packages and Libraries:
pip install numpy
pip install pandas
pip install scikit-learn
pip install xgboost
pip install matplotlib
pip install seaborn
pip install streamlit
pip install pickle
## Installation


```bash
pip install numpy 
pip install pandas 
pip install scikit-learn 
pip install xgboost 
pip install matplotlib 
pip install seaborn 
pip install streamlit 
pip install pickle
pip install scipy
pip install xgboost
pip install imblearn
```
    
## Demo video link

https://www.linkedin.com/posts/rathika-kavitha-nagaraj-954b3219a_here-is-the-demo-video-on-my-next-project-activity-7202979020941004800-1BeG?utm_source=share&utm_medium=member_desktop
