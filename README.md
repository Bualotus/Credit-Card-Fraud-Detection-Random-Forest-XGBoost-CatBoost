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

This section prepare the data is ready before trainning model, start from Data Asssessment to examine data to determine its quality, Data Cleaning for handling missing value and duplicate value, Data Transformation for feature extraction and encoding, Data Reduction for Dropping Columns that are Unique Identifiers and unnecessary data, and Split data set to split data into trainning, validation and test set.
<hr style="border: none; border-top: 5px double #333;">

# **5.Model Trainning**






