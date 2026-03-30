# Student Dropout Risk Prediction Project

## Overview
This project aims to predict the risk of student dropout using a combination of behavior clustering techniques and the XGBoost machine learning model. By analyzing student behavior data, we can identify students who are at a higher risk of dropping out and provide interventions based on the predicted outcomes.

## Features
- Clusters students based on behavior patterns.
- Predicts dropout risk using XGBoost.
- Visualizations for understanding student behavior and predictions.

## Architecture
- **Data Collection**: Gather data on student behavior and academic performance.
- **Data Preprocessing**: Clean and prepare data for analysis.
- **Behavior Clustering**: Use clustering algorithms to segment students into groups based on behavior.
- **Model Training**: Train an XGBoost model to predict dropout risks.
- **Evaluation**: Assess model performance using metrics such as accuracy and F1 score.

## Dependencies
- Python 3.x
- XGBoost
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/sam058/student-risk-prediction.git
   cd student-risk-prediction
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Project Structure
```
student-risk-prediction/
├── data/
├── models/
├── notebooks/
├── scripts/
├── README.md
└── requirements.txt
```

## Step-by-Step Execution Flow
1. **Run Data Collection**: Execute scripts in `scripts/data_collection.py` to gather data.
2. **Data Preprocessing**: Run the preprocessing script to clean and format your data.
3. **Clustering**: Execute the behavior clustering script to group students.
4. **Train Model**: Run `scripts/train_model.py` to train the XGBoost model on the processed data.
5. **Prediction**: Use the trained model to predict dropout risks.

## Usage Instructions
- Run the main execution file to go through the entire flow:
   ```bash
   python main.py
   ```
- Provide dataset path and parameters as needed.

## Model Details
- **Model**: XGBoost
- **Training Data**: Features derived from behavior clustering plus additional student data.
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1 Score.

## Outputs
- Evaluation metrics after model training.
- Visualizations showing the prediction results and behavior clusters.

## Troubleshooting
- Ensure all dependencies are installed correctly.
- Check data paths and formats if errors occur during data loading.
- Debugging information will be logged in the console.

For any further assistance, please open an issue in the GitHub repository or contact the project maintainers.