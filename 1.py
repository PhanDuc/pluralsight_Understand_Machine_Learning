import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
pd.set_option('display.width', 320)
df = pd.read_csv('E:\\Big Data - Data Mining\\pluralsight_Understand_Machine_Learning\\python-understanding-machine-learning\\4-python-understanding-machine-learning-m4-exercise-files\\data\\pima-data.csv')

def plot_corr(df, size):
    """

    :param df: pandas DataFrame
    :param size: vertical and horizontal size of the pilot

    :Display :
        matrix of correlation between column :
            blue - cyan - red - darkred : less to more correlated

    """
    corr = df.corr()
    fig, ax = plt.subplots(figsize = (size, size))
    ax.matshow(corr)
    plt.xticks(range(len(corr.columns)), corr.columns)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.show()

del df['skin']

diabetes_map = {True: 1, False : 0}
df['diabetes'] = df['diabetes'].map(diabetes_map)

# Check true and false ratio
num_true = len(df.loc[df['diabetes'] == True])
num_false = len(df.loc[df['diabetes'] == False])

print("Number of True cases : {0} ({1:2.2f}%)".format(num_true, (num_true / (num_true + num_false)) * 100))
print("Number of false cases : {0} (({1:2.2f}%)".format(num_false, (num_false / (num_true + num_false))* 100))

# Spliting the data
from sklearn.model_selection import train_test_split

feature_col_names = ['num_preg', 'glucose_conc', 'diastolic_bp', 'thickness', 'insulin', 'bmi', 'diab_pred', 'age']
predicted_class_name  = ['diabetes']

X = df[feature_col_names].values
y = df[predicted_class_name].values
split_test_size = 0.3

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state= 42) # Chia theo 1 bo so ngu nhien

print("{0:0.2f}% in training set ".format((len(X_train) / len(df.index)) * 100))
print("{0:0.2f}% in test set ".format((len(X_test) / len(df.index)) * 100))

print("rows missing glucose_conc : {0}".format(len(df.loc[df['glucose_conc'] == 0])))

# Impute data
from sklearn.preprocessing import Imputer

fill_0 = Imputer(missing_values= 0, strategy='mean', axis= 0)

X_train = fill_0.fit_transform(X_train)
X_test = fill_0.fit_transform(X_test)

# Training Naive Bayes
from sklearn.naive_bayes import GaussianNB

nb_model = GaussianNB()
nb_model.fit(X_train, y_train.ravel())

# Performance on Training Data
# preddict values using the training data
nb_predict_train = nb_model.predict(X_train)

# import the performance metric library
from sklearn import metrics

# Accuracy
print("Accuracy: {0:.4f}".format(metrics.accuracy_score(y_train, nb_predict_train)))

# Performance on Testing Data
nb_predict_test = nb_model.predict(X_test)
print("Accuracy: {0:4f}".format(metrics.accuracy_score(y_test, nb_predict_test)))

# Confussion Matrix
print("Confussion Matrix")
print("{0}".format(metrics.confusion_matrix(y_test, nb_predict_test, labels=[1,0])))
print("")

# Random Forest
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(random_state= 42)
rf_model.fit(X_train, y_train.ravel())

# Predict training data : Random Forest dua ra mo hinh giong danh gia du lieu goc da dua vao dung den 98%
rf_predict_train = rf_model.predict(X_train)
print("Accuracy Random Forest: {0:.4f}".format(metrics.accuracy_score(y_train, rf_predict_train)))

# Predict test data
rf_predict_test = rf_model.predict(X_test)
print("Accuracy Random Forest - Test Data: {0:.4f}".format(metrics.accuracy_score(y_test, rf_predict_test)))

# Logistics Regression
from sklearn.linear_model import LogisticRegression

lr_model = LogisticRegression(C = 0.7, random_state= 42)
lr_model.fit(X_train, y_train.ravel())

lr_predict_test = lr_model.predict(X_test)

print("Accuracy: {0:4f}".format(metrics.accuracy_score(y_test, lr_predict_test)))
print(metrics.confusion_matrix(y_test, lr_predict_test, labels=[1, 0]))

C_start = 0.1
C_end = 5
C_inc = 0.1

C_values, recall_scores = [], []

C_val = C_start
best_recall_score = 0

while (C_val < C_end):
    C_values.append(C_val)
    lr_model_loop = LogisticRegression(C = C_val, class_weight="balanced", random_state=42)
    lr_model_loop.fit(X_train, y_train.ravel())
    lr_predict_loop_test = lr_model_loop.predict(X_test)

    recall_score = metrics.recall_score(y_test, lr_predict_loop_test)
    recall_scores.append(recall_score)

    if(recall_score > best_recall_score):
        best_recall_score = recall_score
        best_lr_presict_test = lr_predict_loop_test

    C_val = C_val + C_inc

best_score_C_val = C_values[recall_scores.index(best_recall_score)]
print("lrt max value of {0:.3f} occured at C = {1:.3f}".format(best_recall_score, best_score_C_val))

plt.plot(C_values, recall_scores, "-")
plt.xlabel("C value")
plt.ylabel("recall scores")
plt.show()

lr_model = LogisticRegression(C = best_score_C_val, class_weight="balanced", random_state= 42)