# Data Science Problem Description

## Usage Instructions

### Option 1: Kaggle Competition URL
Simply paste a Kaggle competition URL in this file:
```
https://www.kaggle.com/competitions/playground-series-s5e4/overview
```

The system will automatically:
- Extract competition information
- Parse the problem description
- Determine problem type (regression/classification)
- Set up the appropriate workflow

### Option 2: Custom Problem Description
Write your own problem description below:

## Problem Overview
The dataset for this competition was generated from a deep learning model trained on Podcast Listening Time Prediction data.

## Goal
Predict `Listening_Time_minutes` from given features in train.csv.

## Problem Type
The system automatically determines if this is a **regression** or **classification** problem based on the target variable characteristics.

- **Regression**: Predicts continuous values (e.g., prices, scores, quantities)
- **Classification**: Predicts categorical values (e.g., yes/no, categories, classes)

## Dataset Information
- **Training Data**: Located in `./data/train/train.csv`
- **Test Data**: Located in `./data/test/` (will be created automatically if missing)
- **Target Variable**: `Listening_Time_minutes`
- **Features**: Various podcast and user-related features

## Success Criteria
The model should be optimized for:
1. **Accuracy**: Minimize prediction errors
2. **Generalization**: Perform well on unseen data
3. **Robustness**: Handle various data scenarios
4. **Interpretability**: Provide insights into feature importance

## Expected Deliverables
1. **Cleaned and processed data** with proper naming conventions
2. **Feature engineered dataset** optimized for modeling
3. **Trained model** saved as `./model/model.pkl` or `./model/model.h5`
4. **Comprehensive evaluation** with detailed metrics
5. **Complete solution notebook** with all code and analysis

## Technical Requirements
- Use advanced feature engineering techniques
- Implement cross-validation for robust evaluation
- Perform hyperparameter tuning
- Generate comprehensive evaluation reports
- Follow best practices for data science workflows

## Constraints
- Ensure data safety and security
- Maintain code quality and documentation
- Optimize for both performance and interpretability
- Handle missing data and outliers appropriately 