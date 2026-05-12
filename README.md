# 🩺 Diabetes Risk Prediction & Patient Segmentation

A machine learning project that predicts diabetes risk using clinical patient data from the Pima Indians Diabetes Database. This project includes comprehensive data analysis, model development, and an interactive Streamlit web application for real-time predictions.

## 📋 Project Overview

This project builds and deploys a diabetes risk prediction system using multiple machine learning algorithms. The system analyzes 8 key clinical features to predict the likelihood of diabetes in patients, providing risk stratification (low, moderate, high) with personalized medical recommendations.

**Dataset**: [Pima Indians Diabetes Database (UCI Machine Learning Repository)](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
- **Records**: 768 patients
- **Features**: 8 clinical measurements
- **Target**: Binary classification (Diabetes: Yes/No)

## ✨ Features

- **Interactive Web Application**: Streamlit-based UI for real-time predictions
- **Multiple ML Models**: Logistic Regression, SVM, Decision Tree, Random Forest
- **Comprehensive Analysis**: Data preprocessing, EDA, feature correlation analysis
- **Risk Stratification**: Three-tier risk assessment (Low, Moderate, High)
- **Clinical Recommendations**: Personalized guidance based on risk level
- **Interactive Visualizations**: Confusion matrices, ROC curves, feature analysis
- **Model Comparison**: Performance metrics for all algorithms

## 🚀 Quick Start

### Installation

1. **Clone the repository**:
```bash
git clone https://github.com/darkdevil3610/Diabetes-Risk-Prediction-Patient-Segmentation.git
cd Diabetes-Risk-Prediction-Patient-Segmentation
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Ensure model file exists**:
   - The `model.pkl` file should be in the same directory as `app.py`
   - This file contains the trained Random Forest model and scaler

### Running the Application

Start the Streamlit app:
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

### Running the Jupyter Notebook

For full analysis and model training:
```bash
jupyter notebook Diabetes.ipynb
```

## 📊 Input Features

The model uses the following 8 clinical measurements:

| Feature | Range | Unit |
|---------|-------|------|
| Pregnancies | 0-17 | Count |
| Glucose | 44-199 | mg/dL |
| Blood Pressure | 24-122 | mmHg |
| Skin Thickness | 7-99 | mm |
| Insulin | 14-846 | μU/ml |
| BMI | 10-70 | kg/m² |
| Diabetes Pedigree Function | 0.05-2.50 | Score |
| Age | 21-81 | years |

## 🎯 Risk Classification

### High Risk (≥ 60% probability)
- **Action**: Immediate medical consultation advised
- **Recommendations**: Fasting glucose test, HbA1c screening, lifestyle intervention

### Moderate Risk (35-60% probability)
- **Action**: Monitor blood glucose regularly
- **Recommendations**: Lifestyle changes (diet, exercise)

### Low Risk (< 35% probability)
- **Action**: Maintain healthy lifestyle
- **Recommendations**: Annual check-ups

## 📈 Model Performance

| Model | Accuracy |
|-------|----------|
| Logistic Regression | 0.8796 |
| Support Vector Machine | 0.9444 |
| Decision Tree | 0.9352 |
| **Random Forest** | **0.9722** |

**Best Model**: Random Forest (n_estimators=300) with 97.22% accuracy

## 📁 Project Structure

```
Diabetes-Risk-Prediction-Patient-Segmentation/
├── app.py                    # Streamlit web application
├── Diabetes.ipynb            # Jupyter notebook with analysis & model training
├── model.pkl                 # Serialized trained model + scaler
├── README.md                 # This file
└── requirements.txt          # Python dependencies
```

## 🔧 Key Dependencies

- **pandas**: Data manipulation
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning models and preprocessing
- **matplotlib & seaborn**: Data visualization
- **streamlit**: Web application framework
- **joblib**: Model serialization
- **plotly**: Interactive visualizations
- **imbalanced-learn**: SMOTE-ENN for handling class imbalance

## 📚 Methodology

### Data Preprocessing
1. Handling missing values (0-coded in original dataset)
2. Outlier detection and removal
3. Log transformation for normalization
4. StandardScaler normalization

### Class Imbalance Handling
- SMOTE-ENN (Synthetic Minority Oversampling Technique + Edited Nearest Neighbors)
- Stratified train-test split (80-20)

### Feature Engineering
- Correlation analysis
- PCA dimensionality reduction
- Feature importance evaluation

### Model Training
- Cross-validation (5-10 folds)
- Hyperparameter tuning (Grid Search)
- ROC curve analysis

## 🔍 Example Usage

### Via Web Application:
1. Enter patient clinical data in the sidebar
2. Click **"🔍 Predict Diabetes Risk"**
3. View risk classification and clinical recommendations
4. Review key risk flags and detailed analysis

### Healthy Patient Example:
- Pregnancies: 0
- Glucose: 85 mg/dL
- Blood Pressure: 70 mmHg
- Skin Thickness: 20 mm
- Insulin: 79 μU/ml
- BMI: 21
- Diabetes Pedigree Function: 0.2
- Age: 24 years

**Expected Result**: Low Risk

## ⚠️ Important Disclaimer

**This tool is for educational and research purposes only.** It should not replace professional medical diagnosis. Predictions are based on machine learning models and should always be validated by qualified healthcare professionals.

## 🛠️ Development Notes

### Model Serialization:
The model and scaler are saved together:
```python
joblib.dump({
    "model": model,
    "scaler": scaler
}, "model.pkl")
```

### Probability Calculation:
- Uses `predict_proba()` for calibrated probabilities
- Falls back to `decision_function()` → sigmoid transformation if needed
- Output is clamped between 0.0 and 1.0

## 📊 Visualizations Included

- Class distribution (imbalanced data)
- Correlation heatmap
- Feature boxplots by outcome
- Feature histograms (diabetic vs non-diabetic)
- Scatter plots (feature relationships)
- PCA dimensionality reduction
- Confusion matrices for all models
- ROC curves with AUC scores
- Model accuracy/loss comparisons

## 🤝 Contributing

Feel free to submit issues and enhancement requests!

## 📄 License

This project is open source. Please refer to your organization's policies for usage and distribution.

## 📧 Contact

For questions or suggestions, please open an issue on GitHub.

---

**Last Updated**: May 2026  
**Dataset Source**: UCI Machine Learning Repository  
**Framework**: Streamlit + scikit-learn
