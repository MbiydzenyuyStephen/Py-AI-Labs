import pandas as pd
import seaborn as sns
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import VotingClassifier
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, roc_auc_score

#from google.colab import files
# Upload the file manually if it's on your local machine
#uploaded = files.upload()

#load and inspect the data
data = pd.read_excel("synthetic_telecom_churn_dataset.xlsx", engine='openpyxl')

#to get the first five rows of the dataset
data.head()

#to get the last five rows of the dataset
data.tail()

data.info()

data = data.drop("customer_id" ,axis=1) #drop irrelevant columns

data.describe()

#encoding categorical data(region)
#data["region"].unique() used to get unique values of this cell

data = pd.get_dummies(data,drop_first=True)

# Convert all boolean columns to integers
bool_cols = data.select_dtypes(include='bool').columns
data[bool_cols] = data[bool_cols].astype(int)

data.info()

#Exploring missing values and handling them

#to get the total amount of empty cells for each column
data.isnull().sum()

#analyse data distribution and anomalies

#method to handle outliers
def handle_outliers(data,feature):
  Q1 = data[feature].quantile(0.25)
  Q3 = data[feature].quantile(0.75)

  IQR=Q3-Q1
  upper_limit = Q3 + 1.5 * IQR
  lower_limit = Q1 - 1.5 * IQR

  # Identify outliers.
  outliers = data[(data[feature] < lower_limit) | (data[feature] > upper_limit)]
  print(f"Feature '{feature}' - Number of outliers: {len(outliers)}")

  # Cap the outliers.
  #here we want to give ouliers with high upperlimit = upperlimit and vice versa.
  data[feature] = data[feature].apply(lambda x: lower_limit if x < lower_limit else upper_limit if x > upper_limit else x)

  return data



def clean_numeric_column(data, column):
    data[column] = data[column].replace('[^\d]', '', regex=True)  # Remove non-numeric characters
    data[column] = pd.to_numeric(data[column], errors='coerce')  # Convert to numeric, setting errors to NaN
    return data

numerical_data = ['age', 'income', 'monthly_minutes', 'monthly_data_gb',
                      'support_tickets', 'monthly_bill', 'outstanding_balance']
for feature in numerical_data:
   data = clean_numeric_column(data, feature)
   data = handle_outliers(data,feature)



sns.pairplot(data, hue='age')

#check for correlations between features

#identifying correlation between features
correlation_matrix = data.corr()

sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')


data['churn'].value_counts()

sns.countplot(x='churn', data=data)

#training our model using logistic regression
#preparing the data

year1_data = data.iloc[:2000]
year2_data = data.iloc[2000:4000]
year3_data = data.iloc[4000:5000]

X_year1 = year1_data.drop(columns='churn')
y_year1 = year1_data['churn']

X_year2 = year2_data.drop(columns='churn')
y_year2 = year2_data['churn']

X_year3 = year3_data.drop(columns='churn')
y_year3 = year3_data['churn']


columns = [
    'age',                   # The age of the customer
    'income',                # The income of the customer
    'monthly_minutes',       # Monthly phone usage in minutes
    'monthly_data_gb',      # Monthly data usage in gigabytes
    'support_tickets',       # Number of support tickets raised by the customer
    'monthly_bill',          # Monthly bill amount
    'outstanding_balance',    # Outstanding balance amount
    'region_North',         # Binary feature indicating if the customer is from the North region
    'region_South',         # Binary feature indicating if the customer is from the South region
    'region_West'           # Binary feature indicating if the customer is from the West region
]

# Define a function to train and evaluate the models
def train_and_evaluate(X_train, y_train, X_test, y_test):
    # Handle class imbalance
    X_balanced, y_balanced = SMOTE().fit_resample(X_train, y_train)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_balanced)
    X_test_scaled = scaler.transform(X_test)

    # Convert scaled features back to DataFrame
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=columns)

    # Train models
    logistic_model = LogisticRegression(max_iter=12000, solver="lbfgs")
    decision_tree_model = DecisionTreeClassifier(class_weight='balanced')

    logistic_model.fit(X_train_scaled, y_balanced)
    decision_tree_model.fit(X_train_scaled, y_balanced)

    # Make predictions
    y_pred_logistic = logistic_model.predict(X_test_scaled)
    y_pred_tree = decision_tree_model.predict(X_test_scaled)

    # Print metrics
    print("Logistic Regression Metrics:")
    print(f'Accuracy: {accuracy_score(y_test, y_pred_logistic)}')
    print(f'Precision: {precision_score(y_test, y_pred_logistic)}')
    print(f'Recall: {recall_score(y_test, y_pred_logistic)}')
    print(f'F1 Score: {f1_score(y_test, y_pred_logistic)}')
    print(f'AUC ROC: {roc_auc_score(y_test, y_pred_logistic)}')
    print(classification_report(y_test, y_pred_logistic))

    print("\nDecision Tree Metrics:")
    print(f'Accuracy: {accuracy_score(y_test, y_pred_tree)}')
    print(f'Precision: {precision_score(y_test, y_pred_tree)}')
    print(f'Recall: {recall_score(y_test, y_pred_tree)}')
    print(f'F1 Score: {f1_score(y_test, y_pred_tree)}')
    print(f'AUC ROC: {roc_auc_score(y_test, y_pred_tree)}')
    print(classification_report(y_test, y_pred_tree))
    return  roc_auc_score(y_test, y_pred_logistic)


# Train on Year 1 and test on Year 2
print("Training on Year 1 and testing on Year 2:")
roc_auc_score2 =train_and_evaluate(X_year1, y_year1, X_year2, y_year2)


# Train on Year 2 and test on Year 3
print("\nTraining on Year 2 and testing on Year 3:")
roc_auc_score3=train_and_evaluate(X_year2, y_year2, X_year3, y_year3)

# Investigate Changes Over Time
# You can compare the ROC AUC scores and classification reports for Year 2 and Year 3
print(f"Comparison of ROC AUC Scores: Year 2: {roc_auc_score2}, Year 3: {roc_auc_score3}")

# Additional analysis for concept drift or data shift
if roc_auc_score3 < roc_auc_score2:
    print("Warning: Potential concept drift detected. Model performance has deteriorated from Year 2 to Year 3.")
else:
    print("Model performance is consistent or improved from Year 2 to Year 3.")


X1_balanced, y1_balanced = SMOTE().fit_resample(X_year1, y_year1)
X2_balanced, y2_balanced = SMOTE().fit_resample(X_year2, y_year2)
X3_balanced, y3_balanced = SMOTE().fit_resample(X_year3, y_year3)

# Generate gradually increasing weights for each year's data
weights_1 = np.linspace(1, 2, len(y1_balanced))
weights_2 = np.linspace(2, 3, len(y2_balanced))
weights_3 = np.linspace(3, 3.5, len(y3_balanced))

# Scale features
scaler = StandardScaler()
X_year1_scaled = scaler.fit_transform(X1_balanced)
X_year2_scaled = scaler.transform(X2_balanced)
X_year3_scaled = scaler.transform(X3_balanced)


#  Online learning with SGDClassifier
sgd_model = SGDClassifier(max_iter=1000, tol=1e-3)
# Train initially on Year 1
sgd_model.fit(X_year1_scaled, y1_balanced, sample_weight=weights_1)

# Incremental training on Year 2 using partial_fit
sgd_model.partial_fit(X_year2_scaled, y2_balanced, sample_weight=weights_2)


# Incremental training on Year 3 using partial_fit
sgd_model.partial_fit(X_year3_scaled, y3_balanced, sample_weight=weights_3)

# Evaluate on Year 1
y_pred_year1 = sgd_model.predict(X_year1_scaled)
accuracy_year1 = accuracy_score(y1_balanced, y_pred_year1)

# Evaluate on Year 2
y_pred_year2 = sgd_model.predict(X_year2_scaled)
accuracy_year2 = accuracy_score(y2_balanced, y_pred_year2)

# Evaluate on Year 3
y_pred_year3 = sgd_model.predict(X_year3_scaled)
accuracy_year3 = accuracy_score(y3_balanced, y_pred_year3)

# Print metrics for each year
print(f"Online Learning Model - Year 1 Metrics:")
print(f"Accuracy: {accuracy_year1}")

print(f"Online Learning Model - Year 2 Metrics:")
print(f"Accuracy: {accuracy_year2}")

print(f"Online Learning Model - Year 3 Metrics:")
print(f"Accuracy: {accuracy_year3}")

# Train separate models for year2 and year3
sgd_model_year2 = SGDClassifier(max_iter=1000, tol=1e-3)
sgd_model_year3 = SGDClassifier(max_iter=1000, tol=1e-3)

sgd_model_year2.fit(X_year2_scaled, y2_balanced, sample_weight=weights_2)
sgd_model_year2.partial_fit(X_year3_scaled, y3_balanced, sample_weight=weights_3)

sgd_model_year3.fit(X_year3_scaled, y3_balanced, sample_weight=weights_3)

# Evaluate on Year 3
y_pred_year3 = sgd_model.predict(X_year3_scaled)

# Metrics for Year 3
accuracy = accuracy_score(y3_balanced, y_pred_year3)
precision = precision_score(y3_balanced, y_pred_year3)
recall = recall_score(y3_balanced, y_pred_year3)
f1 = f1_score(y3_balanced, y_pred_year3)
roc_auc = roc_auc_score(y3_balanced, y_pred_year3)

print(f"Online Learning Model - Year 3 Metrics:")
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print(f"AUC ROC: {roc_auc}")
print(classification_report(y3_balanced, y_pred_year3))


# Ensemble Model Training: Voting Classifier
ensemble_model = VotingClassifier(estimators=[
    ('sgd_year1', sgd_model),
    ('sgd_year2', sgd_model_year2),
    ('sgd_year3', sgd_model_year3),
], voting='hard') # 'soft' voting for probability averaging

x_1_2 = np.concatenate([X_year1_scaled, X_year2_scaled])
y_1_2 = np.concatenate([y1_balanced, y2_balanced])
weights_1_2 = np.concatenate([weights_1, weights_2])

ensemble_model.fit(x_1_2, y_1_2)

#Predict on Year 3 using the ensemble model
y2_pred_ensemble = ensemble_model.predict(X_year2_scaled)
y3_pred_ensemble = ensemble_model.predict(X_year3_scaled)

# Evaluate Model Performance on Year 2
ensemble_accuracy2 = accuracy_score(y2_balanced, y2_pred_ensemble)
ensemble_precision = precision_score(y2_balanced, y2_pred_ensemble)
ensemble_recall = recall_score(y2_balanced, y2_pred_ensemble)
ensemble_f1 = f1_score(y2_balanced, y2_pred_ensemble)

print(f"Ensemble Model - Accuracy: {ensemble_accuracy2}")
print(f"Ensemble Model - Precision: {ensemble_precision}")
print(f"Ensemble Model - Recall: {ensemble_recall}")
print(f"Ensemble Model - F1 Score: {ensemble_f1}")

# Evaluate Model Performance on Year 3
ensemble_accuracy = accuracy_score(y3_balanced, y3_pred_ensemble)
ensemble_precision = precision_score(y3_balanced, y3_pred_ensemble)
ensemble_recall = recall_score(y3_balanced, y3_pred_ensemble)
ensemble_f1 = f1_score(y3_balanced, y3_pred_ensemble)

print(f"Ensemble Model - Accuracy: {ensemble_accuracy}")
print(f"Ensemble Model - Precision: {ensemble_precision}")
print(f"Ensemble Model - Recall: {ensemble_recall}")
print(f"Ensemble Model - F1 Score: {ensemble_f1}")