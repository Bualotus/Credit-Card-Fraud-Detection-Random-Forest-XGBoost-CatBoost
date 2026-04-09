# Credit-Card-Fraud-Detection-Random-Forest-XGBoost-CatBoost🔍
Machine learning models for Credit card fraud detection using  Random Forest, XGBoost, CatBoost  to compare performance and give practical recommendations and implications to accurately identifying fraudulent transactions and minimising false alarms with tuning the hyperparameter and the decision threshold.

# **Project Overview**
This project aims to build machine learning models for Credit card fraud detection using supervised learning models in different algorithms including 

1) Random Forest
2) XGBoost
3) CatBoost 

to compare performance and give practical recommendations and implications. This project presents how to address the imbalance in data, stop financial loss from fraudulent activities by accurately identifying fraudulent transactions and minimising false alarms with tuning the hyperparameter and the decision threshold.

---
Machine Learning Model Building Pipeline including:

<img width="1923" height="475" alt="Credit Card Fraud Detection End-to End Process" src="https://github.com/user-attachments/assets/d101e467-3a87-4e6c-8b99-c930b0d98592" />


**I developed this project to practice data analytics skill and learning machine learning. If you have any suggestions or feedback, I would be very thank you for your help to improve my project and skills 😊.**

---
# **Table of Content**




---
# **1.Problem Statement**
Financial Fraud is still a significant global challenge and is increasing. Because advanced artificial intelligence is being used in fraud tactics, reshaping the contours of financial crime. While Artificial intelligence and machine learning have become significantly part of fraud detection to predict and indicate the fraudulent activity pattern. Credit card fraud is a type of financial fraud that involves the unauthorised use of a credit card to make fraudulent transactions. There are many machine learning algorithms to build a model for Credit card fraud detection as a detection control and develop for real-time fraud detection.

This project aims to build machine learning models for Credit card fraud detection using supervised learning models in different algorithms to compare performance and give practical recommendations and implications.


<hr style="border: none; border-top: 5px double #333;">

# **2.Data Collection**
The dataset used in this study is “Credit Card Transactions Fraud Detection Dataset “ from Kaggle, containing legitimate and fraudulent transactions between 1 January 2019 and 31st Dec 2020, containing  21 features such as transaction date and time, merchan, category, amount and customer information. The target variable, is_fraud, indicates whether a transaction is fraudulent (1) or not (0). In the case of the real-world situation, all entries have been anonymised to protect customer privacy.

<hr style="border: none; border-top: 5px double #333;">

# **3.Data Exploration**
This section will present Data overview, Descriptive Statistic, and Vistualisation to highlight the relationship between feature and target class and imblance data.
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

This section prepares the data is ready before trainning model, start from Data Asssessment to examine data to determine its quality, Data Cleaning for handling missing value and duplicate value, Data Transformation for feature extraction and encoding, Data Reduction for Dropping Columns that are Unique Identifiers and unnecessary data, and Split data set to split data into trainning, validation and test set.
<hr style="border: none; border-top: 5px double #333;">

# **5.Model Trainning**
This section presents the trainning of baseline and optimised model including;
**Baseline Model**
  **1. Random Forest (Baseline)**
    A random forest is an ensemble of decision trees, where each tree is trained on a random subset of data and features. The final prediction is made by aggregating (voting or averaging)   the results from all trees. It is advantageous to reduce overfitting with high accuracy and also provide feature importance scores for model optimisation. 
  **2. XGBoost (Baseline)**
    XGBoost is a gradient boosting algorithm, which is designed to be highly efficient, flexible and portable.  It is an optimized gradient boosting algorithm that combines multiple weak    models into a stronger, high-performance model by using decision trees as base learners, building them sequentially so each tree corrects errors from the previous one and it is known as boosting. Moreover, It features parallel processing for faster training on large datasets and allows parameter customisation to optimize performance for specific problems.
  **3. CatBoost (Baseline)**
    CatBoost is an advanced gradient-boosting library specifically designed to address the challenges of handling categorical data in machine learning. CatBoost is an open-source technology that has become quite popular quickly because it can produce high-performance models without requiring a lot of data preprocessing including missing value and encoding.

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
  **6. XGBoost (Optimised)** 
    I used 'scale_pos_weight'as a hyperparameter used to address class imbalance.
<hr style="border: none; border-top: 5px double #333;">

# **6. Model Evaluation**
This section compare the model performance including Recall, Precision, F1 score, and  PR-AUC (Area Under the Precision-Recall Curve).
The result as below;

  **Recall🎣** 
  <img width="927" height="556" alt="image" src="https://github.com/user-attachments/assets/228e66f5-32fd-4f09-9ef6-9af252336acd" />
  
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

**Post-tuning the decision threshold**
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

---

### The Economic Impact Formula🧮 
We define our success by the following profit-based metric:

> **Net Savings** = (Revenue Saved by Catching Fraud) − (Operational Cost from False Alarms) − (Money Lost from Missed Fraud)

$$Net\ Savings = (TP \times R_{TP}) + (TN \times R_{TP}) - (FP \times C_{FP}) - (FN \times C_{FN})$$



### Variable Definitions🔍 
* **$R_{TP}$ (Reward per Caught Fraud):** The value saved by blocking a fraudulent charge. I assume operational savings is 50.00 dolloars per transaction.
* **$R_{TN}$ (Reward per successful Payment):** the 2% fee represents the Reward per Successful Payment
* **$C_{FP}$ (Cost per False Alarm):** The operational cost per alert (e.g., customer support time, SMS verification). I assume that operational cost is 10.00 dolloars per alert.
* **$C_{FN}$ (Cost per Missed Fraud):** The actual financial loss from the stolen funds and chargebacks. The cost is calculated from the average monetary value of a fraudulent transaction.

### Objective💡 

The goal of post-tuning is to find the Optimal Threshold—the probability threshold that maximises our Net Savings by using TunedThresholdClassifierCV .

---

  

  
 






