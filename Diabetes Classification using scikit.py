# diabetes_classification.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
column_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 
                'DiabetesPedigreeFunction', 'Age', 'Outcome']


df = pd.read_csv(url, names=column_names)


print("Dataset Preview:")
print(df.head())

X = df.drop('Outcome', axis=1)  # Features
y = df['Outcome']  # Target (0 = no diabetes, 1 = diabetes)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\nScaled Data (First 5 rows):")
print(X_scaled[:5])


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


print("\nTraining set shape:", X_train.shape)
print("Testing set shape:", X_test.shape)


model = LogisticRegression(random_state=42)


model.fit(X_train, y_train)


y_pred = model.predict(X_test)


print("\nPredicted Outcomes (First 10):", y_pred[:10])
print("Actual Outcomes (First 10):", y_test[:10].values)


accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy * 100:.2f}%")

conf_matrix = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(conf_matrix)

class_report = classification_report(y_test, y_pred)
print("\nClassification Report:")
print(class_report)

plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['No Diabetes', 'Diabetes'], 
            yticklabels=['No Diabetes', 'Diabetes'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
