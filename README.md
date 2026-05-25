ead# Explainable AI for Mental Health Classification

An Explainable AI project focused on mental health treatment prediction, model transparency, fairness analysis, and bias mitigation. This project demonstrates how machine learning can be used responsibly in sensitive healthcare-related contexts by combining predictive modelling with interpretability tools such as SHAP and LIME.

## Project Overview

Mental health AI systems can support early intervention and decision-making, but they also raise concerns around transparency, fairness, bias, privacy, and patient safety.

This project builds a mental health treatment prediction pipeline that:

- Generates a synthetic mental-health-style dataset
- Trains and compares multiple machine learning classifiers
- Selects the best-performing model using evaluation metrics
- Explains model predictions using SHAP and LIME
- Analyses fairness across gender groups
- Applies bias mitigation using sample reweighting
- Produces visual outputs and an ethical summary report

## Key Features

- Synthetic mental health dataset generation
- Data preprocessing and categorical encoding
- Model training and evaluation
- Multiple classifier comparison
- SHAP global and local explainability
- LIME local prediction explanations
- Gender-based fairness analysis
- Bias mitigation using sample reweighting
- Ethical report generation
- Visual outputs for performance, explainability, and fairness

## Technologies Used

- Python
- pandas
- NumPy
- scikit-learn
- Matplotlib
- Seaborn
- SHAP
- LIME

## Machine Learning Models

The project trains and compares the following classifiers:

- Random Forest Classifier
- Gradient Boosting Classifier
- Logistic Regression
- Decision Tree Classifier

The best model is selected based on F1 score.

## Dataset

The project uses a synthetic mental-health-style dataset with 1,000 samples. The dataset includes demographic, workplace, and mental health-related features.

Example features include:

- Age
- Gender
- Country
- Self-employment status
- Family history of mental health conditions
- Work interference
- Company size
- Remote work status
- Tech company status
- Mental health benefits
- Care options
- Wellness program availability
- Help-seeking support

The target variable is `treatment`, which represents whether the individual is likely to seek mental health treatment.

The synthetic dataset includes designed correlations, such as higher treatment probability for individuals with family history, work interference, and access to mental health benefits. A small gender-related bias is also introduced to support fairness analysis.

## Project Workflow

### 1. Data Generation

The project begins by generating a synthetic dataset that represents mental health-related survey data. This allows the project to demonstrate responsible AI methods without using real sensitive health data.

### 2. Exploratory Data Analysis

The script explores the target distribution and age distribution, then saves visualizations.

Generated file: `01_data_exploration.png`

### 3. Data Preprocessing

Categorical variables are encoded using label encoding, and features are scaled using standardization.

Preprocessing includes:

- Separating features and target variable
- Encoding categorical columns
- Scaling numerical features
- Splitting the data into training and testing sets

### 4. Model Training

Multiple classifiers are trained and evaluated using:

- Accuracy
- Precision
- Recall
- F1 score
- ROC-AUC score

Generated file: `02_model_comparison.png`

### 5. Model Evaluation

The best-performing model is evaluated using a confusion matrix.

Generated file: `03_confusion_matrix.png`

If the best model supports feature importance, the project also generates a feature importance plot.

Generated file: `04_feature_importance.png`

### 6. SHAP Explainability

SHAP is used to explain how features influence model predictions. The project generates both global and detailed SHAP visualizations.

Generated files:

- `05_shap_summary.png`
- `06_shap_detailed.png`
- `07_shap_force.png`

SHAP helps answer questions such as:

- Which features are most important overall?
- How does each feature influence the prediction?
- Why did the model make a specific prediction?

### 7. LIME Explainability

LIME is used to generate local explanations for individual predictions. This helps show why the model predicted treatment or no treatment for specific samples.

Generated file: `08_lime_explanations.png`

### 8. Fairness Analysis

The project evaluates fairness across gender groups using metrics such as:

- Positive prediction rate
- True positive rate
- False positive rate
- Precision

Generated file: `09_fairness_analysis.png`

### 9. Bias Mitigation

A simple sample reweighting strategy is applied to reduce gender-based prediction imbalance. The project compares fairness before and after mitigation.

Generated file: `10_bias_mitigation.png`

### 10. Ethical Report

The project generates a short ethical summary report describing fairness metrics, bias mitigation results, and key ethical considerations.

Generated file: `ethical_report.txt`

## Output Files

| File | Description |
|---|---|
| `01_data_exploration.png` | Target and age distribution plots |
| `02_model_comparison.png` | Model performance comparison |
| `03_confusion_matrix.png` | Confusion matrix for the best model |
| `04_feature_importance.png` | Feature importance chart |
| `05_shap_summary.png` | SHAP global feature importance |
| `06_shap_detailed.png` | Detailed SHAP feature impact plot |
| `07_shap_force.png` | SHAP force plot for an individual prediction |
| `08_lime_explanations.png` | LIME explanations for sample predictions |
| `09_fairness_analysis.png` | Fairness metrics across gender groups |
| `10_bias_mitigation.png` | Effect of reweighting on fairness |
| `ethical_report.txt` | Summary of fairness and ethical findings |

## Installation

Clone the repository:

    git clone https://github.com/your-username/your-repository-name.git
    cd your-repository-name

Install the required dependencies:

    pip install numpy pandas matplotlib seaborn scikit-learn shap lime

Or create a `requirements.txt` file with:

    numpy
    pandas
    matplotlib
    seaborn
    scikit-learn
    shap
    lime

Then install:

    pip install -r requirements.txt

## How to Run

Run the main Python script:

    python mental_health.py

The script will:

1. Generate the dataset
2. Train and evaluate models
3. Run SHAP explanations
4. Run LIME explanations
5. Analyse fairness
6. Apply bias mitigation
7. Generate plots and an ethics report

## Example Results

The model comparison stage identifies the best-performing model based on F1 score. In the project presentation, Gradient Boosting achieved the strongest F1 score among the tested models.

The explainability analysis found that important predictors included:

- Family history
- Work interference
- Mental health benefits
- Care options
- Workplace support factors

The fairness analysis showed that different demographic groups can experience different prediction rates and error rates, highlighting the need for fairness monitoring in mental health AI systems.

## Ethical Considerations

Mental health AI systems involve sensitive data and high-stakes decisions. This project highlights several ethical concerns.

### Transparency

Users and clinicians should be able to understand why a model makes a prediction. SHAP and LIME help make black-box models more interpretable.

### Fairness

Models should be tested across demographic groups to detect unequal outcomes. Fairness should be continuously monitored.

### Privacy

Mental health data is highly sensitive. Real-world systems should follow privacy laws and use strong data protection methods.

### Human Oversight

AI should support, not replace, mental health professionals. High-risk decisions should involve human review.

### Safety

Mental health AI systems must include clear escalation processes for crisis situations or uncertain predictions.

## Limitations

This project is designed as a research and demonstration system. It has several limitations:

- The dataset is synthetic and does not represent real clinical data.
- Fairness analysis is limited to gender-based comparisons.
- The model does not use longitudinal mental health data.
- Cultural and regional differences are not fully captured.
- The project is not suitable for real clinical deployment without validation.
- Bias mitigation is simple and would need stronger methods in production.

## Future Improvements

Possible future extensions include:

- Using real anonymized and ethically sourced mental health datasets
- Adding text, voice, or behavioral data for multi-modal analysis
- Expanding fairness analysis across age, country, and workplace groups
- Adding more advanced bias mitigation techniques
- Building an interactive dashboard for model explanations
- Integrating clinician feedback loops
- Testing the system across different cultural contexts
- Adding privacy-preserving techniques such as federated learning or differential privacy

## Project Structure

    .
    ├── mental_health.py
    ├── README.md
    ├── 01_data_exploration.png
    ├── 02_model_comparison.png
    ├── 03_confusion_matrix.png
    ├── 04_feature_importance.png
    ├── 05_shap_summary.png
    ├── 06_shap_detailed.png
    ├── 07_shap_force.png
    ├── 08_lime_explanations.png
    ├── 09_fairness_analysis.png
    ├── 10_bias_mitigation.png
    └── ethical_report.txt

## Suggested Repository Description

Explainable AI project for mental health treatment prediction using SHAP, LIME, fairness analysis, and bias mitigation techniques.

## Suggested Topics

- machine-learning
- explainable-ai
- mental-health
- shap
- lime
- fairness
- bias-mitigation
- healthcare-ai
- responsible-ai
- python
- scikit-learn

## Disclaimer

This project is for educational and research purposes only. It is not a medical device and should not be used for clinical diagnosis, treatment decisions, or mental health assessment. Any real-world mental health AI system would require clinical validation, regulatory review, privacy safeguards, and human oversight.

## Author

Summen Zahid

COM772 - Emerging and Advanced Topics in AI
