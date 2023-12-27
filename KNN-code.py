#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


data = pd.read_csv("diabetes.csv")
data.head()


# In[3]:


data.isnull().any()


# In[4]:


data.describe().T


# In[5]:


zero_columns = (data == 0).any()
zero_columns = zero_columns[zero_columns].index.tolist()
print(zero_columns)


# In[6]:


data_copy = data.copy(deep=True)
data_copy[zero_columns] = data_copy[zero_columns].replace(0, np.NaN)
data_copy.isnull().sum()


# In[7]:


p = data.hist(figsize = (20,20))


# In[12]:


data_copy['Pregnancies'].fillna(data_copy['Pregnancies'].mean(), inplace = True)
data_copy['Glucose'].fillna(data_copy['Glucose'].mean(), inplace = True)
data_copy['BloodPressure'].fillna(data_copy['BloodPressure'].median(), inplace = True)
data_copy['SkinThickness'].fillna(data_copy['SkinThickness'].median(), inplace = True)
data_copy['Insulin'].fillna(data_copy['Insulin'].median(), inplace = True)
data_copy['BMI'].fillna(data_copy['BMI'].mean(), inplace = True)
data_copy['Pedigree'].fillna(data_copy['Pedigree'].median(), inplace = True)
data_copy['Age'].fillna(data_copy['Age'].median(), inplace = True)
data_copy['Outcome'].fillna(data_copy['Outcome'].median(), inplace = True)


# In[13]:


p = data.hist(figsize = (20,20))


# In[16]:


pip install missingno


# In[18]:


import missingno as msno
p = msno.bar(data)


# In[21]:


p=data.Outcome.value_counts().plot(kind="bar")


# In[22]:


import seaborn as sns
p=sns.pairplot(data_copy, hue = 'Outcome')


# In[23]:


import matplotlib.pyplot as plt
plt.figure(figsize=(12,10))  # on this line I just set the size of figure to 12 by 10.
p=sns.heatmap(data.corr(), annot=True,cmap ='RdYlGn')  # seaborn has very simple solution for heatmap


# In[30]:


from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_scaled = sc_X.fit_transform(data.drop(["Outcome"], axis=1))
columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'Pedigree', 'Age']
X = pd.DataFrame(X_scaled, columns=columns)
X.head()


# In[31]:


y =data_copy.Outcome


# In[32]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 42, stratify=y)


# In[33]:


from sklearn.neighbors import KNeighborsClassifier

train_scores = []
test_scores = []

for i in range(1,15):
    knn = KNeighborsClassifier(i)
    knn.fit(X_train, y_train)
    train_scores.append(knn.score(X_train, y_train))
    test_scores.append(knn.score(X_test, y_test))


# In[34]:


max_test_score =max(test_scores)


# In[35]:


test_score_index = [i for i, v in enumerate(test_scores) if v== max_test_score]

print('Max test score {} % and k = {}'.format(max_test_score*100,list(map(lambda x: x+1, test_score_index))))


# In[36]:


plt.figure(figsize=(12,5))
p = sns.lineplot(range(1,15),train_scores,marker='*',label='Train Score')
p = sns.lineplot(range(1,15),test_scores,marker='o',label='Test Score')


# In[37]:


#Setup a knn classifier with k neighbors
knn = KNeighborsClassifier(11)

knn.fit(X_train,y_train)
knn.score(X_test,y_test)


# In[38]:


from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, fbeta_score
y_pred = knn.predict(X_test)

cnf_matrix = confusion_matrix(y_test, y_pred)


# In[39]:


p = sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')


# In[40]:


def model_evaluation(y_test, y_pred, model_name):
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    f2 = fbeta_score(y_test, y_pred, beta = 2.0)

    results = pd.DataFrame([[model_name, acc, prec, rec, f1, f2]], 
                       columns = ["Model", "Accuracy", "Precision", "Recall",
                                 "F1 SCore", "F2 Score"])
    results = results.sort_values(["Precision", "Recall", "F2 Score"], ascending = False)
    return results



model_evaluation(y_test, y_pred, "KNN")


# In[41]:


# Alternate way
from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))


# In[51]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_curve, auc
import matplotlib.pyplot as plt


# In[53]:


df = pd.read_csv('diabetes.csv')


# In[57]:


X = df.drop('Outcome', axis=1)  # Features
y = df['Outcome']  # Target variable


# In[58]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[59]:


# Create a logistic regression model
logreg = LogisticRegression(random_state=42)


# In[61]:


logreg = LogisticRegression(random_state=42, max_iter=1000)


# In[62]:


logreg.fit(X_train, y_train)


# In[63]:


# Make predictions on the test set
y_pred = logreg.predict(X_test)


# In[64]:


# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# In[65]:


# Calculate the ROC curve and AUC
y_pred_proba = logreg.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)


# In[66]:


# Plot the ROC curve
plt.figure(figsize=(8, 8))
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line representing random chance
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve for Diabetes Prediction')
plt.legend(loc='lower right')
plt.show()


# In[ ]:




