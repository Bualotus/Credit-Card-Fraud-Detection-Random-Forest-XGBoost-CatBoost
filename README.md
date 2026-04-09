# Credit-Card-Fraud-Detection-Random-Forest-XGBoost-CatBoost💳
Machine learning models for Credit card fraud detection using  Random Forest, XGBoost, CatBoost  to compare performance and give practical recommendations and implications to accurately identifying fraudulent transactions and minimising false alarms with tuning the hyperparameter and the decision threshold.

# **Project Overview**

> **Objective:**

This project aims to build machine learning models for Credit card fraud detection using supervised learning models in different algorithms including 
1) Random Forest
2) XGBoost
3) CatBoost

to compare performance and give practical recommendations and implications.

> **Methodology:**

This project presents how to address the imbalance in data, stop financial loss from fraudulent activities by accurately identifying fraudulent transactions and minimising false alarms with tuning the hyperparameter and the decision threshold based on the business strategies.

Dataset from Kaggle. 
The dataset used in this study is the [Credit Card Transactions Fraud Detection Dataset](https://www.kaggle.com/datasets/kartik2112/fraud-detection) 
Note: Due to file size limits, the raw dataset is not included in this repository. Please download it directly from the Kaggle link provided above.


> **Key Achievement:**

Optimized CatBoost with custom threshold (0.09) as the best model to identify **90% of fraud cases** (Recall 0.9) and **projected to save ~$1,313k per year** (based on $525k savings from the test set period),   significantly reducing financial loss while maintaining optimal customer experience.

---
Machine Learning Model Building Pipeline including:

<img width="1923" height="475" alt="Credit Card Fraud Detection End-to End Process" src="https://github.com/user-attachments/assets/d101e467-3a87-4e6c-8b99-c930b0d98592" />


**I developed this project to practice data analytics skill and learning machine learning. If you have any suggestions or feedback, I would be very thank you for your help to improve my project and skills 😊.**

---
# 📑 Table of Contents
* [📌 1. Problem Statement](#1problem-statement)
* [📊 2. Data Collection](#2data-collection)
* [🔍 3. Data Exploration](#3data-exploration)
* [🛠️ 4. Data Preprocessing](#4data-preprocessing)
* [🧪 5. Model Training](#5model-training)
* [🏆 6. Model Evaluation](#6-model-evaluation)
    * [Model Performance Comparison](#model-performance-comparison)
    * [Post-tuning the decision threshold](#post-tuning-the-decision-threshold)
    * [Final Evaluation](#final-evaluation)
* [🚀 7. Model Deployment](#7-model-deployment)
* [📚 Reference](#reference)


---
## 📂 Repository Structure
* **`README.md`**: Provides a executive summary and business results.
* **`Credit_Card_Fraud_Detection_Project.ipynb`**: Contains the complete end-to-end Python code
* **`requirements.txt`**: List of necessary libraries to run the project.
* **`.gitignore`**: Specifies which files and directories should be ignored by Git.
---

# **1.Problem Statement**
Financial Fraud is still a significant global challenge and is increasing. Because advanced artificial intelligence is being used in fraud tactics, reshaping the contours of financial crime. While Artificial intelligence and machine learning have become significantly part of fraud detection to predict and indicate the fraudulent activity pattern. Credit card fraud is a type of financial fraud that involves the unauthorised use of a credit card to make fraudulent transactions. There are many machine learning algorithms to build a model for Credit card fraud detection as a detection control and develop for real-time fraud detection.

This project aims to build machine learning models for Credit card fraud detection using supervised learning models in different algorithms to compare performance and give practical recommendations and implications.


<hr style="border: none; border-top: 5px double #333;">

# **2.Data Collection**
The dataset used in this study is “Credit Card Transactions Fraud Detection Dataset “ from Kaggle, containing legitimate and fraudulent transactions between 1 January 2019 and 31st Dec 2020, containing  21 features such as transaction date and time, merchan, category, amount and customer information. The target variable, is_fraud, indicates whether a transaction is fraudulent (1) or not (0). In the case of the real-world situation, all entries have been anonymised to protect customer privacy.

<hr style="border: none; border-top: 5px double #333;">

# **3.Data Exploration**
This section will present Data overview, Descriptive Statistic, and Visualization to highlight the relationship between feature and target class and imblanced data.
After explore data, I will prepare the features as follows

| No. | Feature | Description and Strategy for Data Preprocessing |
|:---:|:---|:---|
| 1 | **trans_date_trans_time** | Timestamp of the transaction. <br> *Strategy:* Extract temporal features like hour of the day, day of the week, or month. |
| 2 | **cc_num** | Credit card number of the customer. <br> *Strategy:* Drop column due to high cardinality and uniqueness. |
| 3 | **merchant** | Name of the merchant. <br> *Strategy:* Drop column due to high cardinality and uniqueness. |
| 4 | **category** | Category of the transaction (e.g., entertainment, food_dining). <br> *Strategy:* Use *Label Encoding* for Random Forest and XGBoost. |
| 5 | **amt** | Transaction amount (numerical value). <br> *Strategy:* Keep as a numerical feature. |
| 6 | **first, last** | First and last name of the cardholder. <br> *Strategy:* Drop column due to high cardinality and uniqueness. |
| 7 | **gender** | Gender of the cardholder. <br> *Strategy:* Use *Label Encoding* for Random Forest and XGBoost. |
| 8 | **street, city, state, zip** | Address details of the cardholder. <br> *Strategy:* Drop column due to high cardinality. |
| 9 | **lat, long** | Latitude and Longitude of the cardholder. <br> *Strategy:* Create new feature *"Distance"* by calculating distance between cardholder and merchant. |
| 10 | **city_pop** | Population of the cardholder's city. <br> *Strategy:* Keep as a numerical feature. |
| 11 | **job** | Job title/profession of the cardholder. <br> *Strategy:* Drop column due to high cardinality. |
| 12 | **dob** | Date of birth of the cardholder. <br> *Strategy:* Create new feature *"Age"* by calculating from the transaction date. |
| 13 | **trans_num** | Unique identifier for the transaction. <br> *Strategy:* Drop column due to high cardinality. |
| 14 | **unix_time** | Transaction time in Unix format. <br> *Strategy:* Drop column due to high cardinality. |
| 15 | **merch_lat, merch_long** | Latitude and Longitude of the merchant. <br> *Strategy:* Use to calculate the *"Distance"* feature. |
| 16 | **is_fraud** | **Target Variable**: 1 if fraudulent, 0 otherwise. |

<hr style="border: none; border-top: 5px double #333;">

# **4.Data Preprocessing**

This section prepares the data is ready before training model, start from Data Assessment to examine data to determine its quality, Data Cleaning for handling missing value and duplicate value, Data Transformation for feature extraction and encoding, Data Reduction for Dropping Columns that are Unique Identifiers and unnecessary data, and Split data set to split data into training, validation and test set.
<hr style="border: none; border-top: 5px double #333;">

# **5.Model Training**
This section presents the training of baseline and optimised model including;
### **Baseline Model**
  **1. Random Forest (Baseline)**
    A random forest is an ensemble of decision trees, where each tree is trained on a random subset of data and features. The final prediction is made by aggregating (voting or averaging)   the results from all trees. It is advantageous to reduce overfitting with high accuracy and also provide feature importance scores for model optimisation. 
    
  **2. XGBoost (Baseline)**
    XGBoost is a gradient boosting algorithm, which is designed to be highly efficient, flexible and portable.  It is an optimized gradient boosting algorithm that combines multiple weak    models into a stronger, high-performance model by using decision trees as base learners, building them sequentially so each tree corrects errors from the previous one and it is known as boosting. Moreover, It features parallel processing for faster training on large datasets and allows parameter customisation to optimize performance for specific problems.
    
  **3. CatBoost (Baseline)**
    CatBoost is an advanced gradient-boosting library specifically designed to address the challenges of handling categorical data in machine learning. CatBoost is an open-source technology that has become quite popular quickly because it can produce high-performance models without requiring a lot of data preprocessing including missing value and encoding.

### **Optimised Model**

After training the model, I improve model performance and address imbalance data by Feature Important and Stratified K-Fold and Hyperparameter Tuning. Feature Importance can measures how much each feature contributes to the model’s prediction accuracy. It helps in identifying the most influential input variables, improving performance, interpretability and computational efficiency. For hyperparameter tunning, RandomizedSearchCV is a machine-learning technique used to optimize a model's hyperparameters by performing a random search over a specified parameter grid. I started from
    
   - Define Model: Choose the machine learning algorithm you want to optimize
   - Create the Parameter Grid: Specify the parameters and the range of values you want to search over.
   - Instantiate RandomizedSearchCV: Provide the model, parameter grid, and other settings like the number of iterations, cross-       validation strategy, etc.
   - Fit the Model: Run the random search on the training data.
   - Evaluate the Best Model: Check the performance of the best-found parameters.
     
Therefore, this section present the optimised model as below;

  **4. Random Forest (Optimised)**
    Apart from performing Feature Important and Stratified K-Fold and Hyperparameter Tuning. I use 'class_weight' to address class imbalance by assigning different levels of importance to each class during training. It helps the model avoid bias toward the majority class and improves performance on minority classes.
    
  **5. XGBoost (Optimised)**
    I used 'scale_pos_weight'as a hyperparameter used to address class imbalance.
    
  **6. CatBoost (Optimised)** 
    I used 'scale_pos_weight'as a hyperparameter used to address class imbalance.
<hr style="border: none; border-top: 5px double #333;">

# **6. Model Evaluation**
This section present model evaluation to select the best model performance. Then, I perform tunning threshold to improve model, which align a classifier’s predictions with specific business objectives.

## **Model Performance Comparison**
This section compare the model performance including Recall, Precision, F1 score, and  PR-AUC (Area Under the Precision-Recall Curve).
The result as below;

  **Recall🎣** 
  
  <img width="995" height="595" alt="image" src="https://github.com/user-attachments/assets/d97482ae-c662-4321-ae93-70c4cb84c7f7" />

  
  **Precision🎯**
  
  <img width="932" height="555" alt="image" src="https://github.com/user-attachments/assets/86ae4086-881e-4ffb-8ac6-f8c032f5799c" />

  **F1-Score⚖️**
  
  <img width="928" height="551" alt="image" src="https://github.com/user-attachments/assets/d8dbee60-f605-454f-8ebe-5346e5cd89fc" />

  **PR-AUC📈**
  
  <img width="1746" height="522" alt="image" src="https://github.com/user-attachments/assets/8f8294f9-7a22-4094-ba7b-a9f24fde54a2" />

  After comparing the model performance, I selected the optimised model of Catboost as the best-performing model for credit card fraud detection according to the observation as below;
  - CatBoost (Optimized) emerged as the top performer, achieving a recall of 0.789. This means the model successfully identified nearly 80% of all fraudulent attempts, providing the most   robust protection against financial risk.
  -  CatBoost (Optimised) achieved a high precision of 0.906. This indicates that when the model flags a transaction as fraud, it is correct over 90% of the time.
  -  CatBoost (Optimized) reached the highest F1-score of 0.843. This confirms that it offers the best statistical balance between fraud detection and customer experience.
  -  The chart of the CatBoost optimised model stays closer to the top-right corner than any other model. It maintains a Precision near 1.0 even as Recall approaches 0.8. This indicates that the model can catch the vast majority of fraud without increasing false alerts.

## **Post-tuning the decision threshold**
Then, I perform threshold Tuning to align a classifier’s predictions with specific business objectives when false positives and false negatives carry different costs like fraud detection where missing fraud is worse than a false alarm. Threshold tuning is trying different cut points on the probability output, rather than accepting the default 0.5 probability cutoff. When you set the cutoff high, you make fewer positive calls and reduce false positives. If you set it low you catch more positives but invite more false alarms. The goal is pick the cutoff that minimizes the harm you care about.

According to the confusion matrix;
| | Predict Negative | Predict Positive |
|---|:---:|:---:|
| **Actual Negative** | **TN** (True Negative) | **FP** (False Positive) |
| **Actual Positive** | **FN** (False Negative) | **TP** (True Positive) |     

**Where:**
* **$TN$ (True Negative):** Legitimate transactions correctly identified ✅
* **$FP$ (False Positives):** Legitimate transactions wrongly flagged (leading to operational investigation costs). (False Alarm)⚠️
* **$FN$ (False Negatives):** Fraudulent transactions missed by the model (direct financial loss).(Missed Fraud)❌
* **$TP$ (True Positives):** Fraudulent transactions correctly identified and blocked. (Caught Fraud)💰


I performed threshold tuning to align with a business strategy which prioritizes achieving the highest net savings based on our underlying business assumptions. Therefore, **I consider the cost and benefit from FP (False Alarm), FN (Missed Fraud), and TP(Caught Fraud).**


**The Economic Impact Formula🧮**
We define our success by the following profit-based metric:

> **Net Savings** = (Revenue Saved by Catching Fraud) − (Operational Cost from False Alarms) − (Money Lost from Missed Fraud)

$$Net\ Savings = (TP \times R_{TP}) + (TN \times R_{TP}) - (FP \times C_{FP}) - (FN \times C_{FN})$$



**Variable Definitions🔍** 
* **$R_{TP}$ (Reward per Caught Fraud):** The value saved by blocking a fraudulent charge. I assume operational savings is 50.00 dollars per transaction.
* **$R_{TN}$ (Reward per successful Payment):** the 2% fee represents the Reward per Successful Payment
* **$C_{FP}$ (Cost per False Alarm):** The operational cost per alert (e.g., customer support time, SMS verification). I assume that operational cost is 10.00 dollars per alert.
* **$C_{FN}$ (Cost per Missed Fraud):** The actual financial loss from the stolen funds and chargebacks. The cost is calculated from the average monetary value of a fraudulent transaction.

**Objective💡** 

The goal of post-tuning is to find the Optimal Threshold—the probability threshold that maximises our Net Savings by using TunedThresholdClassifierCV .

**Result⚡** 

After perform tunning threshold, I found that the best threshold is 0.09 to generate maximum net saving.


## **Final Evaluation**


<img width="421" height="332" alt="image" src="https://github.com/user-attachments/assets/55d567d2-18ff-453d-b7d7-f76aebd031c9" />


After I have selected the Catboost optimised model as the best performing model. I perform tuning threshold based on the business assumption and found the best threshold at 0.09. By moving the threshold from 0.50 down to 0.09, we can increase the benefit by $42k from the same dataset. ( from $482,895.77 ( baseline) to $525,154.31 (tuning threshold))  Moreover, the model tends to be highly sensitive, with the recall reaching 0.9. It means we can identify 90% of all fraud cases. While the precision is 0.57. Because the cost of a missed fraud (FN) is much higher than the cost of checking a good customer (FP). Therefore, the model prioritises to save the money from those extra caught frauds. Moreover, ROC AUC and Average Precision reach to 0.9974 and 0.9039, respectively. These scores confirm the high performance of model. 

However, the optimised threshold can vary based on the business strategy. Therefore, we should reconsider and monitor the assumption of the model regulary. 

<hr style="border: none; border-top: 5px double #333;">

# **7. Model Deployment**

This select include Deployment Strategy and Model Monitoring & Maintenance to ensure the fraud detection system must be implemented and monitored continuously.
  
<hr style="border: none; border-top: 5px double #333;">
  
 # **Reference**

 Ashington. (2025). ML system design for fraud detection in real-time. Medium. https://ashington.medium.com/ml-system-design-for-fraud-detection-in-real-time-cef7c5526899

Chandrasekaran, C. (n.d.). EDA for a classification problem. Deepnote. https://deepnote.com/app/charan-chandrasekaran/EDA-for-Classification-problem-df3d55a6-09d3-48ec-8aca-d039a4b5ff5a

Coursera. (2025). Data preprocessing steps: How to prepare data for machine learning and analytics. https://www.coursera.org/articles/data-preprocessing-steps

GeeksforGeeks. (2022). Haversine formula to find the distance between two points on a sphere. https://www.geeksforgeeks.org/haversine-formula-to-find-distance-between-two-points-on-a-sphere/

GeeksforGeeks. (2025). Label encoding of datasets in python. https://www.geeksforgeeks.org/ml-label-encoding-of-datasets-in-python/

GeeksforGeeks. (2026). Random Forest algorithm in machine learning. https://www.geeksforgeeks.org/random-forest-algorithm-in-machine-learning/

GeeksforGeeks. (2025). What is feature engineering? https://www.geeksforgeeks.org/what-is-feature-engineering/

GeeksforGeeks. (2026). XGBoost. https://www.geeksforgeeks.org/xgboost/

Olaniyan, B. (2025). Sklearn vs. Imblearn pipeline: Preventing data leakage in imbalanced datasets. Medium. https://medium.com/@banjiolaniyan123/sklearn-vs-imblearn-pipeline-preventing-data-leakage-in-imbalanced-datasets-b4272484e985

Pykes, K. (2025). Data preprocessing: A complete guide with Python examples. DataCamp. https://www.datacamp.com/blog/data-preprocessing

Sanjay V. (2024). What is EDA? Medium. https://medium.com/@sanjayskumar4010/what-is-eda-1abdae6409bf

Scikit-learn. (n.d.). Metrics and scoring: Quantifying the quality of predictions. https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-string-names

Scikit-learn. (n.d.). Post-tuning the decision threshold for cost-sensitive learning. https://scikit-learn.org/stable/auto_examples/model_selection/plot_cost_sensitive_learning.html

Shenoy, K. (2020). Credit card transactions fraud detection dataset [Data set]. Kaggle. https://www.kaggle.com/datasets/kartik2112/fraud-detection

Stack Overflow. (2016). How to tune parameters in random forest using scikit-learn. https://stackoverflow.com/questions/36107820/how-to-tune-parameters-in-random-forest-using-scikit-learn

Talent500. (n.d.). Feature engineering: Creating new features to enhance model performance. https://talent500.com/blog/feature-engineering-creating-new-features-to-enhance-model-performance/

Verzi, V. (2025). Understanding model evaluation metrics in fraud detection: Beyond accuracy. Medium. https://medium.com/@valeria.verzi1/understanding-model-evaluation-metrics-in-fraud-detection-beyond-accuracy-52b224ac0418

<hr style="border: none; border-top: 5px double #333;">




