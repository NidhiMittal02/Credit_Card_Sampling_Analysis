# 📊 Sampling Techniques and Model Performance Analysis

## Credit Card Fraud Detection

### 📌 Project Overview
Credit card fraud detection is a critical application of machine learning due to the highly imbalanced nature of transaction data, where fraudulent cases occur far less frequently than legitimate ones. Traditional machine learning models tend to become biased toward the majority class, leading to poor fraud detection performance.


---

### 📂 Dataset   [[Credit Card Dataset](https://github.com/AnjulaMehto/Sampling_Assignment/blob/main/Creditcard_data.csv)]
- **Dataset Name:** `Creditcard_data.csv`  
- **Target Variable:** `Class`  
  - `0` → Non-fraudulent transaction  
  - `1` → Fraudulent transaction  
- **Dataset Nature:** Highly imbalanced  
- **Source:** Publicly available credit card transaction dataset  

---

### ⚙️ Methodology

#### 1️⃣ Data Preprocessing
- The target column (`Class`) was separated from the feature set.  
- All features were standardized using `StandardScaler` to ensure uniform scaling.  
- To handle class imbalance, **SMOTE** (Synthetic Minority Oversampling Technique) was applied, generating synthetic fraud samples and balancing the dataset.  

#### 2️⃣ Sampling Techniques
After balancing the dataset, five different sampling strategies were applied to create multiple subsets for experimentation:

| Sampling ID  | Sampling Technique       | Description                                 |
|-------------|------------------------|--------------------------------------------|
| Sampling1   | Simple Random Sampling   | Random selection of samples from the dataset |
| Sampling2   | Systematic Sampling     | Selection of every k-th record             |
| Sampling3   | Stratified Sampling     | Maintains class proportion in samples      |
| Sampling4   | Cluster Sampling        | Dataset divided into clusters; one cluster selected |
| Sampling5   | Bootstrap Sampling      | Sampling with replacement                   |

These techniques help evaluate how data selection affects model accuracy.

#### 3️⃣ Machine Learning Models
The following five supervised machine learning models were trained and evaluated:

| Model ID | Model Name                   |
|----------|-------------------------------|
| M1       | Logistic Regression           |
| M2       | Decision Tree                 |
| M3       | Random Forest                 |
| M4       | K-Nearest Neighbors (KNN)     |
| M5       | Support Vector Machine (SVM)  |

#### 4️⃣ Model Evaluation Strategy
- **Cross-Validation:** 5-Fold Cross-Validation was used to ensure robust and unbiased evaluation.  
- **Performance Metric:** Accuracy  
- The mean accuracy across the five folds was recorded for each model–sampling combination.

---

### 📈 Accuracy Comparison Table

| Model          | Sampling1 | Sampling2 | Sampling3 | Sampling4 | Sampling5 |
|----------------|-----------|-----------|-----------|-----------|-----------|
| M1_Logistic    | 91.2      | 85.3      | 92.2      | 93.11     | 94.2      |
| M2_DecisionTree| 97.4      | 87.24     | 98.2      | 81.97     | 98.1      |
| M3_RandomForest| 99.7      | 98.69     | 99.8      | 96.72     | 99.4      |
| M4_KNN         | 92.4      | 78.11     | 92.6      | 90.49     | 92.9      |
| M5_SVM         | 97.7      | 95.09     | 97.8      | 98.03     | 98.8      |

---

### ✅ Insights
- **Random Forest** consistently gave the highest accuracy across almost all sampling techniques.  
- **Bootstrap and Stratified sampling** tend to improve model performance compared to Systematic or Cluster sampling.  
- Sampling choice plays a crucial role in model performance, especially for **imbalanced datasets** like credit card fraud detection.  

---

### 📌 Conclusion
This project demonstrates the importance of combining **data balancing techniques (like SMOTE)** with various sampling strategies to improve machine learning model performance on highly imbalanced datasets.  

