# A Comparative Framework for Diabetes Prediction Using Balanced Machine Learning and Deep Learning Models

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.5+-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Abstract

Diabetes mellitus, one of the most common chronic diseases, requires early diagnosis to reduce complications and improve patient outcomes. Machine learning (ML) and deep learning (DL) can predict diabetes. Class imbalance remains a major prediction issue. We compare balanced machine learning and deep learning models for diabetes prediction in this study. The BRFSS 2015 diabetes dataset was used. Machine learning models used the Synthetic Minority Oversampling Technique (SMOTE) to address class imbalance, while deep learning models used the original class distribution. In this study, four machine learning models namely logistic regression (LR), random forest (RF), naive bayes (NB), and extreme gradient boosting (XGBoost), together with three deep learning models, namely artificial neural network (ANN), convolutional neural network (CNN), and recurrent neural network (RNN) were assessed. Two feature configurations were tested: all features and AnyHealthcare removed to assess its diabetes prediction performance. Precision, recall, accuracy, and F1-score assessed performance. In experiments, ANN had the highest accuracy of 84.99%, CNN 84.97%, and XGBoost 81.70%. Deep learning models were slightly more accurate, but XGBoost had a better precision-recall balance. A minimal performance degradation after feature removal indicated strong feature redundancy in the dataset. The framework assesses balanced ML and DL diabetes prediction methods and their model robustness and feature contribution.

## Table of Contents

- [Background](#background)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Authors](#authors)
- [Citation](#citation)
- [License](#license)

## Background

Diabetes mellitus is a chronic metabolic disorder characterized by elevated blood glucose levels resulting from inadequate insulin production or ineffective insulin utilization. According to global health reports, diabetes continues to affect millions of individuals and contributes significantly to mortality and healthcare expenditures. Early detection of diabetes is therefore critical for effective disease management and prevention of severe complications.

Recent advancements in artificial intelligence have enabled healthcare practitioners to leverage machine learning and deep learning techniques for disease prediction. These techniques can analyze large volumes of health-related data and identify hidden patterns associated with diabetes risk. However, healthcare datasets frequently suffer from class imbalance, where the number of healthy individuals significantly exceeds the number of diabetic cases. Such imbalance often leads to biased predictive models.

### Scale of the Problem

The Centers for Disease Control and Prevention has indicated that as of 2018:
- **34.2 million** Americans have diabetes
- **88 million** have prediabetes
- **1 in 5** diabetics are unaware of their risk
- **8 in 10** prediabetics are unaware of their risk

Type II diabetes is the most common form and its prevalence varies by age, education, income, location, race, and other social determinants of health. Much of the burden of the disease falls on those of lower socioeconomic status. Diabetes also places a massive burden on the economy, with diagnosed diabetes costs of approximately **$327 billion** annually and total costs with undiagnosed diabetes and prediabetes approaching **$400 billion** per year.

## Dataset

### Source

The Behavioral Risk Factor Surveillance System (BRFSS) is a health-related telephone survey collected annually by the Centers for Disease Control and Prevention (CDC). Each year, the survey collects responses from over 400,000 Americans on health-related risk behaviors, chronic health conditions, and the use of preventative services. It has been conducted every year since 1984.

For this project, a CSV of the dataset available on Kaggle for the year 2015 was used. The original dataset contains responses from 441,455 individuals with 330 features. These features are either questions directly asked of participants or calculated variables based on individual participant responses.

### Dataset Specifications

| Attribute | Description |
|-----------|-------------|
| Source | CDC BRFSS 2015 |
| Original Records | 441,455 |
| Cleaned Records | 253,680 |
| Original Features | 330 |
| Selected Features | 21 |
| Classes | 0 = No Diabetes, 1 = Prediabetes, 2 = Diabetes |

### Selected Features

Based on identified risk factors from diabetes research literature, the following features were selected from the BRFSS 2015 dataset. Feature descriptions reference the BRFSS 2015 Codebook.

#### Response Variable

| Variable | Description |
|----------|-------------|
| Diabetes_012 | 0 = No Diabetes, 1 = Prediabetes, 2 = Diabetes |

#### Independent Variables

**Cardiovascular Risk Factors**

| Variable | Description |
|----------|-------------|
| HighBP | Adults who have been told they have high blood pressure |
| HighChol | Ever told blood cholesterol is high |
| CholCheck | Cholesterol check within past five years |

**Physical Health**

| Variable | Description |
|----------|-------------|
| BMI | Body Mass Index |

**Lifestyle Factors**

| Variable | Description |
|----------|-------------|
| Smoker | Smoked at least 100 cigarettes in lifetime |
| PhysActivity | Physical activity during past 30 days |
| Fruits | Consume fruit 1 or more times per day |
| Veggies | Consume vegetables 1 or more times per day |
| HvyAlcoholConsump | Heavy drinkers (men >14 drinks/week, women >7 drinks/week) |

**Other Chronic Conditions**

| Variable | Description |
|----------|-------------|
| Stroke | Ever told you had a stroke |
| HeartDiseaseorAttack | Coronary heart disease or myocardial infarction |

**Healthcare Access**

| Variable | Description |
|----------|-------------|
| AnyHealthcare | Any health care coverage |

**General and Mental Health**

| Variable | Description |
|----------|-------------|
| GenHlth | General health status |
| MentHlth | Days mental health not good (past 30 days) |
| PhysHlth | Days physical health not good (past 30 days) |
| DiffWalk | Serious difficulty walking or climbing stairs |

**Demographics**

| Variable | Description |
|----------|-------------|
| Sex | Sex of respondent |
| Age | Age category of respondent |
| Education | Highest grade or year of school completed |
| Income | Annual household income category |

### Important Risk Factors

Research in the field has identified the following as important risk factors for diabetes and other chronic illnesses:

- Blood pressure (high)
- Cholesterol (high)
- Smoking
- Diabetes
- Obesity
- Age
- Sex
- Race
- Diet
- Exercise
- Alcohol consumption
- Body Mass Index (BMI)
- Household Income
- Marital Status
- Sleep
- Time since last checkup
- Education
- Health care coverage
- Mental Health

### References

- BRFSS 2015 Codebook: [https://www.cdc.gov/brfss/annual_data/2015/pdf/codebook15_llcp.pdf](https://www.cdc.gov/brfss/annual_data/2015/pdf/codebook15_llcp.pdf)
- Xie, Z., et al. "Building Risk Prediction Models for Type 2 Diabetes Using Machine Learning Techniques." Preventing Chronic Disease, 2019. [https://www.cdc.gov/pcd/issues/2019/19_0109.htm](https://www.cdc.gov/pcd/issues/2019/19_0109.htm)

## Methodology

### Research Framework

This study proposes a diabetes prediction framework using balanced machine learning and deep learning models. The workflow consists of data preprocessing, class balancing using SMOTE, model training, feature removal analysis, and performance evaluation. Four machine learning models (logistic regression, random forest, naive bayes, and XGBoost) and three deep learning models (ANN, CNN, and RNN) were evaluated under two scenarios: using all features and using a reduced feature set obtained by removing the AnyHealthcare feature.

### Experimental Design

Two experimental scenarios were implemented:

1. **All Features**: All 21 available features were used for training and evaluation
2. **Feature Removal**: The AnyHealthcare attribute was removed to evaluate its contribution to prediction performance

### Data Balancing

- SMOTE was applied exclusively to machine learning models
- Deep learning models were trained using the original class distribution

### Models Evaluated

| Category | Models |
|----------|--------|
| Machine Learning | Logistic Regression, Naive Bayes, Random Forest, XGBoost |
| Deep Learning | Artificial Neural Network (ANN), Convolutional Neural Network (CNN), Recurrent Neural Network (RNN) |

### Model Configurations

| # | Model | Configuration |
|---|-------|---------------|
| 1 | Logistic Regression | max_iter = 800, multi_class = 'multinomial' |
| 2 | Naive Bayes | Gaussian Naive Bayes with default parameters |
| 3 | Random Forest | n_estimators = 90, max_depth = None |
| 4 | XGBoost | n_estimators = 110, learning_rate = 0.13, max_depth = 11 |
| 5 | ANN | Dense(32, relu), Dense(3, softmax), Adam(0.001), Early Stopping |
| 6 | CNN | Conv1D(32, relu), Dense(3, softmax), Adam(0.001), Early Stopping |
| 7 | RNN | SimpleRNN(32, tanh), Dense(3, softmax), Adam(0.001), Early Stopping |

### Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-Score

## Results

### Experiment 1: All Features

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| ANN | 84.99% | 0.8044 | 0.8499 | 0.8081 |
| CNN | 84.97% | 0.8043 | 0.8497 | 0.8101 |
| XGBoost | 84.50% | 0.8038 | 0.8450 | 0.8170 |
| RNN | 84.33% | 0.7112 | 0.8433 | 0.7716 |
| Random Forest | 82.92% | 0.8025 | 0.8292 | 0.8145 |
| Naive Bayes | 66.83% | 0.8338 | 0.6683 | 0.7265 |
| Logistic Regression | 63.99% | 0.8518 | 0.6399 | 0.7172 |

### Experiment 2: Feature Removal (AnyHealthcare Removed)

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| ANN | 84.95% | 0.8049 | 0.8495 | 0.8059 |
| CNN | 84.89% | 0.8034 | 0.8489 | 0.8052 |
| XGBoost | 84.51% | 0.8038 | 0.8451 | 0.8170 |
| RNN | 84.24% | 0.7097 | 0.8424 | 0.7704 |
| Random Forest | 82.93% | 0.8026 | 0.8293 | 0.8146 |
| Naive Bayes | 66.98% | 0.8343 | 0.6698 | 0.7287 |
| Logistic Regression | 64.06% | 0.8517 | 0.6406 | 0.7176 |

### Key Findings

1. ANN achieved the highest classification accuracy among all evaluated models
2. XGBoost demonstrated the best F1-score, indicating superior balance between precision and recall
3. Feature removal analysis showed minimal performance degradation (<0.05%), suggesting strong feature redundancy
4. Deep learning models achieved competitive performance despite training on imbalanced data

## Installation

### Prerequisites

- Python 3.8 or higher
- Git

### Clone the Repository

```bash
git clone https://github.com/Botumraksa/diabetes-prediction-framework.git
cd diabetes-prediction-framework
```

### Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Run Jupyter Notebooks

```bash
jupyter notebook
```

Navigate to `notebooks/03_model_training/` to access all model training notebooks.

### Notebook Execution Order

| Order | Notebook | Model | SMOTE | Feature Removal |
|-------|----------|-------|-------|-----------------|
| 1 | `01_logistic_regression_smote.ipynb` | Logistic Regression | Yes | No |
| 2 | `01_logistic_regression_feature_removal.ipynb` | Logistic Regression | Yes | Yes |
| 3 | `02_xgboost_smote.ipynb` | XGBoost | Yes | No |
| 4 | `02_xgboost_feature_removal.ipynb` | XGBoost | Yes | Yes |
| 5 | `03_random_forest_smote.ipynb` | Random Forest | Yes | No |
| 6 | `03_random_forest_feature_removal.ipynb` | Random Forest | Yes | Yes |
| 7 | `04_naive_bayes_smote.ipynb` | Naive Bayes | Yes | No |
| 8 | `04_naive_bayes_feature_removal.ipynb` | Naive Bayes | Yes | Yes |
| 9 | `05_ann_all_features.ipynb` | ANN | No | No |
| 10 | `05_ann_feature_removal.ipynb` | ANN | No | Yes |
| 11 | `06_cnn_all_features.ipynb` | CNN | No | No |
| 12 | `06_cnn_feature_removal.ipynb` | CNN | No | Yes |
| 13 | `07_rnn_all_features.ipynb` | RNN | No | No |
| 14 | `07_rnn_feature_removal.ipynb` | RNN | No | Yes |

## Project Structure

```
diabetes-prediction-framework/
│
├── notebooks/                          # Jupyter notebooks
│   └── 03_model_training/              # Model training notebooks
│       ├── 01_logistic_regression_smote.ipynb
│       ├── 01_logistic_regression_feature_removal.ipynb
│       ├── 02_xgboost_smote.ipynb
│       ├── 02_xgboost_feature_removal.ipynb
│       ├── 03_random_forest_smote.ipynb
│       ├── 03_random_forest_feature_removal.ipynb
│       ├── 04_naive_bayes_smote.ipynb
│       ├── 04_naive_bayes_feature_removal.ipynb
│       ├── 05_ann_all_features.ipynb
│       ├── 05_ann_feature_removal.ipynb
│       ├── 06_cnn_all_features.ipynb
│       ├── 06_cnn_feature_removal.ipynb
│       ├── 07_rnn_all_features.ipynb
│       └── 07_rnn_feature_removal.ipynb
│
├── paper/                              # Research paper
│   └── Diabetes prediction_revised.doc
│
├── models/                             # Saved trained models
│   ├── machine_learning/
│   └── deep_learning/
│
├── results/                            # Results and figures
│   ├── figures/
│   └── tables/
│
├── data/                               # Dataset files
├── src/                                # Source code
├── scripts/                            # Utility scripts
├── requirements.txt                    # Python dependencies
├── .gitignore                          # Git ignore rules
├── LICENSE                             # MIT License
└── README.md                           # Project documentation
```

## Authors

| Order | Name | Role | Affiliation | Email |
|-------|------|------|-------------|-------|
| 1 | Pakrigna Long | Professor | Department of Applied Mathematics and Statistics, Institute of Technology of Cambodia | long.pakrigna@itc.edu.kh |
| 2 | Botumraksa Choub | Data Science Student | Department of Applied Mathematics and Statistics, Institute of Technology of Cambodia | botum.raksa@gmail.com |
| 3 | Kimlong Ngin | Professor | Institute of Information Technology, University of Heng Samrin Thbongkhmum | nginkimlong@gmail.com |

## Citation

If you use this work in your research, please cite:

```bibtex
@article{long2026diabetes,
  title={A Comparative Framework for Diabetes Prediction Using Balanced Machine Learning and Deep Learning Models},
  author={Long, Pakrigna and Choub, Botumraksa and Ngin, Kimlong},
  year={2026},
  institution={Institute of Technology of Cambodia}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Centers for Disease Control and Prevention (CDC) for providing the BRFSS 2015 dataset
- Institute of Technology of Cambodia for research support
- The open-source community for the tools and libraries used in this project

## Contact

For questions or collaboration opportunities, please contact:

**Pakrigna Long** (Professor)
- Email: long.pakrigna@itc.edu.kh
- Department: Applied Mathematics and Statistics
- Institution: Institute of Technology of Cambodia
- Address: Phnom Penh, Cambodia

**Botumraksa Choub** (Data Science Student)
- Email: botum.raksa@gmail.com
- Department: Applied Mathematics and Statistics
- Institution: Institute of Technology of Cambodia
```
