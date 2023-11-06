#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer

# Step 1: Load your data from the Excel file
data = pd.read_excel('training (2) (1).xlsx')

# Remove rows with missing values in the 'input' column
data = data.dropna(subset=['input'])

# Assuming 'input' is your feature column, and 'Classification' is your target variable.
X_train = data['input'].values
y_train = data['Classification'].values

# Handle missing values by replacing them with an empty string
X_train = ["" if pd.isnull(x) else x for x in X_train]

# Step 2: Convert mathematical expressions to numerical features using CountVectorizer
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(X_train)

# Step 3: Initialize and train the SVM
clf = svm.SVC()
clf.fit(X_train, y_train)

# Step 4: Get the support vectors
support_vectors = clf.support_vectors_

# Step 5: Study the support vectors
# You can now analyze the support vectors to understand their influence on the decision boundary.

# Optional: Evaluate the SVM on a test set (if available)
# X_test = ...
# y_test = ...
# y_pred = clf.predict(X_test)
# ... (evaluation metrics)

# Optional: Save the trained SVM model for future use
# import joblib
# joblib.dump(clf, 'svm_model.pkl')
# After fitting the SVM model
print("SVM trained successfully")

# After getting the support vectors
print("Support vectors:", support_vectors)


# In[2]:


import pandas as pd

# Load the test data
test_data = pd.read_excel('testing (2) (1).xlsx')

# Assuming 'Equation' is your feature column, 'output' is the output column.
X_test = test_data['Equation'].values
y_test = (test_data['output'] > 3.5).astype(int)  # Convert output to binary classification

# Handle missing values by replacing them with an empty string
X_test = ["" if pd.isnull(x) else x for x in X_test]

# Convert mathematical expressions to numerical features using CountVectorizer
X_test = vectorizer.transform(X_test)

# Evaluate the SVM on the test set
y_pred = clf.predict(X_test)

# Calculate accuracy
accuracy = (y_pred == y_test).mean()
print(f'Accuracy: {accuracy}')


# In[ ]:




