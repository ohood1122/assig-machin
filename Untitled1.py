#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


import sklearn.svm as svm
housing_data = pd.read_csv('Diabetes.csv')
housing_data.head()


# In[3]:


import pandas as pd


# In[4]:


diabetes_data = pd.read_csv('diabetes.csv')


# In[5]:


# Check for null values in the dataset
print(housing_data.isnull().sum())


# In[6]:


plt.figure(1)
sns.heatmap(housing_data.corr())
plt.title('Correlation On Diabetes Dataset')


# In[7]:


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import pandas as pd


# In[8]:


target_column = 'Outcome'  # Replace with the actual column name


# In[9]:


# Extract features (X) and target variable (y)
X = diabetes_data.drop(target_column, axis=1)
y = diabetes_data[target_column]


# In[10]:


# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[11]:


classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)


# In[12]:


# Make predictions on the test set
y_pred = classifier.predict(X_test)


# In[13]:


# Generate the confusion matrix
cm = confusion_matrix(y_test, y_pred)


# In[14]:


# Visualize the confusion matrix as a heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Predicted 0', 'Predicted 1'], yticklabels=['Actual 0', 'Actual 1'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()


# In[15]:


# Generate the classification report
report = classification_report(y_test, y_pred)
print("\nClassification Report:")
print(report)


# In[16]:


from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split


# In[17]:


target_column = 'Outcome' 


# In[18]:


X = diabetes_data.drop(target_column, axis=1)
y = diabetes_data[target_column]


# In[19]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[20]:


classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)


# In[21]:


# Make predictions on the test set
y_pred = classifier.predict(X_test)


# In[22]:


# Assuming you have probability estimates (predicted probabilities) for the positive class
y_pred_prob = classifier.predict_proba(X_test)[:, 1]


# In[23]:


# Calculate the FPR, TPR, and thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)


# In[24]:


# Plot the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()


# In[25]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes


# In[26]:


# Load the Diabetes dataset
diabetes_data = load_diabetes()
X = diabetes_data.data
y = diabetes_data.target


# In[27]:


# Splitting the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[28]:


# Creating and training the Decision Tree model
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)


# In[29]:


# Making predictions on the test set
y_pred = dt_model.predict(X_test)


# In[30]:


# Calculating classification metrics
# Note: Since this is a regression task, Decision Tree may not be the best choice, but we can still use classification metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')  # For multiclass classification
recall = recall_score(y_test, y_pred, average='weighted')  # For multiclass classification
f1 = f1_score(y_test, y_pred, average='weighted')  # For multiclass classification


# In[31]:


# Printing the results
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)


# In[ ]:




