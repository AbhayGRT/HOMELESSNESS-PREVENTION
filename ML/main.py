

# In[54]:


import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeRegressor  # Change here
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score


# In[55]:


# Load the dataset from the CSV file
data = pd.read_csv('med-insurance.csv')


# In[56]:


data.head(10)


# In[57]:


data.drop(columns=['region'],inplace=True)


# In[58]:


data.head(10)


# In[59]:


data.describe()


# In[60]:


data.info()


# In[61]:


data.isnull().sum()


# In[62]:




# # Separate features and target variable
# X = data.drop('expenses', axis=1)
# y = data['expenses']

# # Splitting the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Define preprocessing for numerical and categorical features
# numerical_features = ['age', 'bmi', 'children', 'Salary']
# categorical_features = ['sex', 'smoker', 'region']

# # Create transformers
# numeric_transformer = Pipeline(steps=[
#     ('scaler', StandardScaler())
# ])

# categorical_transformer = Pipeline(steps=[
#     ('onehot', OneHotEncoder())
# ])

# # Combine transformers
# preprocessor = ColumnTransformer(
#     transformers=[
#         ('num', numeric_transformer, numerical_features),
#         ('cat', categorical_transformer, categorical_features)
#     ])

# # Append regressor to preprocessing pipeline
# pipeline = Pipeline(steps=[('preprocessor', preprocessor),
#                            ('regressor', DecisionTreeRegressor())])  # Change here

# # Define parameter grid for GridSearchCV
# param_grid = {
#     'regressor__max_depth': [None, 10, 20],  # Only include parameters relevant to Decision Tree
#     'regressor__min_samples_split': [2, 5, 10],
#     'regressor__min_samples_leaf': [1, 2, 4]
# }

# # GridSearchCV
# grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1)
# grid_search.fit(X_train, y_train)

# # Best parameters found by GridSearchCV
# print("Best Parameters:", grid_search.best_params_)

# # Evaluation
# y_pred = grid_search.predict(X_test)
# mse = mean_squared_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)
# print("Mean Squared Error:", mse)
# print("R^2 Score:", r2)
# # Calculate accuracy within a certain percentage threshold
# threshold = 0.3
# y_pred_within_threshold = y_pred * (1 + threshold)
# y_pred_below_threshold = y_pred * (1 - threshold)

# correct_predictions = ((y_test >= y_pred_below_threshold) & (y_test <= y_pred_within_threshold)).sum()
# total_predictions = len(y_test)

# accuracy_within_threshold = correct_predictions / total_predictions
# print("Accuracy within {}% threshold:".format(threshold * 100), accuracy_within_threshold)


# In[63]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the sample data
data = pd.read_csv('med-insurance.csv')

# Define features and target variable
X = data.drop(['Output'], axis=1)
y = data['Output']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing categorical and numerical features
categorical_features = ['sex', 'smoker', 'region']
numerical_features = ['age', 'bmi', 'children', 'expenses', 'Salary', 'Difference']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

# Define the model
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Train the model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:",f1)


# In[64]:


import pandas as pd
import numpy as np

# Load the sample data
data = pd.read_csv('med-insurance.csv')

# Define the number of outliers to introduce
num_outliers = 10

# Define the number of errors to introduce in the 'Output' column
num_errors = 20

# Randomly select rows and modify values to introduce outliers
for _ in range(num_outliers):
    row_index = np.random.randint(0, len(data))  # Select a random row index
    column_to_modify = np.random.choice(['age', 'bmi', 'children', 'expenses', 'Salary', 'Difference'])
    data.loc[row_index, column_to_modify] *= np.random.uniform(2, 10)  # Increase the value by a random factor

# Introduce errors in the 'Output' column
for _ in range(num_errors):
    row_index = np.random.randint(0, len(data))  # Select a random row index
    data.loc[row_index, 'Output'] = 1 - data.loc[row_index, 'Output']  # Flip the value (0 to 1 or 1 to 0)

# Print modified data to verify the changes
print(data)


# In[65]:


# Save the modified data to a new CSV file
data.to_csv('modified_sample_data2.csv', index=False)

print("Modified data saved to 'modified_sample_data2.csv'.")


# In[66]:


data = pd.read_csv('modified_sample_data2.csv')

# Define features and target variable
X = data.drop(['Output'], axis=1)
y = data['Output']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing categorical and numerical features
categorical_features = ['sex', 'smoker', 'region']
numerical_features = ['age', 'bmi', 'children', 'expenses', 'Salary', 'Difference']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

# Define the model
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Train the model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:",f1)


# In[67]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Load the modified sample data
data = pd.read_csv('modified_sample_data.csv')

# Encode categorical variables
encoder = LabelEncoder()
data['sex'] = encoder.fit_transform(data['sex'])
data['smoker'] = encoder.fit_transform(data['smoker'])
data['region'] = encoder.fit_transform(data['region'])

# Define features and target variable
X = data.drop(['Output'], axis=1)
y = data['Output']

# Train a RandomForestClassifier for the confusion matrix
model = RandomForestClassifier(random_state=42)
model.fit(X, y)
y_pred = model.predict(X)

# Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y, y_pred), annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


# In[68]:


# Boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(data=data.drop(['sex', 'smoker', 'region'], axis=1))
plt.title('Box Plot')
plt.xticks(rotation=45)
plt.show()


# In[69]:



# Histogram
plt.figure(figsize=(10, 6))
sns.histplot(data['age'], bins=10, kde=True, color='orange')
plt.title('Histogram of Age')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()


# In[70]:



# Pairplot
sns.pairplot(data, hue='Output', palette='husl')
plt.suptitle('Pairplot', y=1.02)
plt.show()


# In[71]:


# Bar Plot
plt.figure(figsize=(8, 6))
sns.countplot(data=data, x='sex', hue='Output', palette='Set2')
plt.title('Count Plot of Sex')
plt.xlabel('Sex')
plt.ylabel('Count')
plt.show()


# In[72]:


# Correlation Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()


# In[ ]:



import pickle

# Assuming the provided code is already executed and the model is trained

# Save the trained model to a pickle file
with open('trained_model.pkl', 'wb') as file:
    pickle.dump(model,file)
