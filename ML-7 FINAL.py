#!/usr/bin/env python
# coding: utf-8

# In[3]:


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


# In[4]:


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


# In[11]:


import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Load the dataset from the Excel file
df = pd.read_excel("testing (2) (1).xlsx")

# Inspect unique values in 'Equation' and 'Output' columns
print("Unique Values in 'Equation':", df['Equation'].unique())
print("Unique Values in 'output':", df['output'].unique())

# Label encode non-numeric values in 'Equation' and 'Output'
label_encoder = LabelEncoder()
df['Equation'] = label_encoder.fit_transform(df['Equation'])
df['output'] = label_encoder.fit_transform(df['output'])

# Assume 'Equation' and 'Output' are your features, and 'Classification' is the target variable
X = df[['Equation', 'output']]
y = df['Classification']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the SVM classifier
clf = SVC(kernel='linear')
clf.fit(X_train, y_train)

# Use the predict function to get predicted labels for the test set
y_pred = clf.predict(X_test)

# Create a DataFrame to display the results
results_df = pd.DataFrame({'Equation': df.loc[X_test.index, 'Equation'],
                           'Output': df.loc[X_test.index, 'output'],
                           'Actual Classification': y_test,
                           'Predicted Classification': y_pred})

# Display the DataFrame
print(results_df)

# Calculate and print the accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# In[12]:


import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Load the training dataset
train_df = pd.read_excel("training (2) (1).xlsx")

# Load the testing dataset
test_df = pd.read_excel("testing (2) (1).xlsx")

# Inspect unique values in 'input' and 'output' columns in the training dataset
print("Unique Values in 'input':", train_df['input'].unique())
print("Unique Values in 'output':", train_df['output'].unique())

# Label encode non-numeric values in 'input' and 'output' columns in the training dataset
label_encoder_train = LabelEncoder()
train_df['input'] = label_encoder_train.fit_transform(train_df['input'])
train_df['output'] = label_encoder_train.fit_transform(train_df['output'])

# Inspect unique values in 'Equation' and 'output' columns in the testing dataset
print("Unique Values in 'Equation':", test_df['Equation'].unique())
print("Unique Values in 'output':", test_df['output'].unique())

# Label encode non-numeric values in 'Equation' and 'output' columns in the testing dataset
label_encoder_test = LabelEncoder()
test_df['Equation'] = label_encoder_test.fit_transform(test_df['Equation'])
test_df['output'] = label_encoder_test.fit_transform(test_df['output'])

# Assume 'input' and 'output' are your features, and 'Classification' is the target variable in the training dataset
X_train = train_df[['input', 'output']]
y_train = train_df['Classification']

# Assume 'Equation' and 'output' are your features in the testing dataset
X_test = test_df[['Equation', 'output']]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# List of kernel functions to experiment with
kernel_functions = ['linear', 'poly', 'rbf', 'sigmoid']

# Iterate over each kernel function and train the SVM classifier
for kernel_function in kernel_functions:
    # Create and train the SVM classifier
    clf = SVC(kernel=kernel_function)
    clf.fit(X_train, y_train)

    # Use the predict function to get predicted labels for the test set
    y_pred = clf.predict(X_test)

    # Calculate and print the accuracy for each kernel function
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy with {kernel_function} kernel: {accuracy}")


# In[ ]:




