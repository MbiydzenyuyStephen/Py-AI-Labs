import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, recall_score, f1_score,roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler



def clean_data(data):
    if data is not None:
        #step one: checking for non numeric data in  numeric column

        # Convert the column to numeric, setting non-numeric values to NaN
        data['Age'] = pd.to_numeric(data['Age'], errors='coerce')
        data['Fare'] = pd.to_numeric(data['Fare'], errors='coerce')
        data['Parch'] = pd.to_numeric(data['Parch'], errors='coerce')
        data['SibSp'] = pd.to_numeric(data['SibSp'], errors='coerce')

        #step two: handling empty cells

        #fill with overall median age
        data['Age'].fillna(data['Age'].median(), inplace=True)

        #fill with the most common value
        data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)

        #fill with median
        data['Fare'].fillna(data['Fare'].median(), inplace=True)

        #fill with 0
        data['Parch'].fillna(0, inplace=True)
        data['SibSp'].fillna(0, inplace=True)

        return data


#data collection
data = pd.read_csv("Titanic-Dataset.csv")

#data cleaning
data = clean_data(data)

#handling outliers in fare and age
sns.boxplot(data=data, x='Fare')
plt.title("Box Plot for Fare")
plt.show()

sns.boxplot(data=data, x='Age')
plt.title("Box Plot for Age")
plt.show()

columns_to_cap = ['Fare', 'Age']  # List of columns to cap

#removing outliers using capping
for col in columns_to_cap:
    lower_limit = data[col].quantile(0.01)
    upper_limit = data[col].quantile(0.99)
    data[col] = np.where(data[col] < lower_limit, lower_limit, data[col])
    data[col] = np.where(data[col] > upper_limit, upper_limit, data[col])

#data normalization using z
scaler = StandardScaler()
data[columns_to_cap] = scaler.fit_transform(data[columns_to_cap])

#feature engineering
data['family_size'] = data['SibSp'] + data['Parch']
data['title'] = data['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())

#feature selection using corellation analysis with target cell

#encoding categorical features
data = pd.get_dummies(data, columns=['Sex', 'Embarked','title'], drop_first=True)

numeric_columns = data.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])

#calculate the correlation matrix
corr_matrix = numeric_columns.corr()

#plot the correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.show()

#looking at the correlation of features with the target variable 'Survived':
print(corr_matrix['Survived'].sort_values(ascending=False))


#model building
data = data.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin','SibSp','Parch'])


x = data.drop(columns=['Survived'])
y = data['Survived']

# Split the data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
classifier.fit(x_train, y_train)

# Make predictions
y_pred= classifier.predict(x_test)

# Evaluate the model
print("Random Forest Classification Report :")
# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.4f}')

# Recall
recall = recall_score(y_test, y_pred)
print(f'Recall: {recall:.4f}')

# F1 Score
f1 = f1_score(y_test, y_pred)
print(f'F1 Score: {f1:.4f}')

# AUC-ROC
auc_roc = roc_auc_score(y_test, y_pred)
print(f'AUC-ROC: {auc_roc:.4f}')

#classification report
print(classification_report(y_test, y_pred))

c_matrix=confusion_matrix(y_test, y_pred)
# Plot confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(c_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Survived', 'Survived'], yticklabels=['Not Survived', 'Survived'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


