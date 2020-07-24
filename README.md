# Mini Project: Spam-or-Ham SMS Classifer

Classifying messages as Spam or Ham (non-spam) with Scikit-learn and with SMOTE imbalance correction.


## Goal

With data of spam sms messages, use NLTK to create a model to predict whether or sms messages is spam or ham (not spam). Furthermore, evaluate the impacts of an imbalanced sample set and how imbalance correction on training data can affect the accuracy of models.


## Context

In UC Berkeley's Principles and Techniques of Data Science (Data 100) class, students were tasked to use what we have learned to distinguish spam or ham (aka non-spam) emails. This project is a continuation of that assignment as I wanted to see if I improve my model to obtain a higher accuracy. Furthermore, primarily I wanted use this project to gain more practice with the entire data science lifecycle and documenting my findings and approaches. In this project, I will be using a new dataset as specified below and do data cleaning and eda, practice implementing NLTK/CountVectorizer/TFIDF to create a better model, and test the impacts of imbalance correction with SMOTE. 

Ultimately, the purpose of this mini-project is to not only gain practice with the skills learned from class, but to also learn new skills that can build upon my knowledge.


## Resources & Dataset Used

* **Python Version:** 3.7
* **Python Libraries Used:** pandas, numpy, imbalance-learn (imblearn), Scikit-Learn (sklearn), NLTK, seaborn, matplotlib
* **Dataset:** UCI SMS spam collection dataset https://www.kaggle.com/uciml/sms-spam-collection-dataset/


## Data Cleaning

* lowercasing all the sms messages
* removing stopwords
* made new column with character count 

## EDA

* I looked at the distributions of sms message character length between spam and ham messages while also collecting the top 10 most frequent words from each. Below are a few highlights:

![](/images/char_hist.png)
![Frequent Spam Words](/images/spam_frequent.png) ![Frequent Ham Words](/images/ham_frequent.png)


## Model Building

* Split the data into train and tests sets with a test size of 33%.
* Tokenized sms messages to be used for model training
* Transformed training data with count vectorizer and Term frequency-inverse document frequency (TFIDF)

### Imbalance Correction

* Used SMOTE (Synthetic Minority Oversampling Technique) to create a alternative training set for model creation

I tried three different models:
* Logistic Regression - used as baseline
* Random Forest - Known to reduce overfitting and interested in seeing how balancing a dataset would impact model accuracy
* Naive Bayes - Assumes features are independent from each other, I thought this was a good fit. Also, model is recommended for document classification

## Model Performance

Random Forest Model did the best regardless of which dataset the model was trained with. While all models improved, Naive Bayes improved significantly. 

**Trained with Imbalanced Dataset:**
* Logistic Regression: 0.961936 
* Random Forest: 0.974443 
* Naive Bayes: 0.868951

**Trained with Balanced Dataset (SMOTE):**
* Logistic Regression: 0.968461
* Random Forest: 0.978793
* Naive Bayes: 0.933660


## Thanks for checking out my SMS Classification Project!



