import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
df = pd.read_csv('crash.csv')  
print("\nFirst few rows of the dataset:")
print(df.head())
print("\nChecking for missing values:")
print(df.isnull().sum())
df.dropna(inplace=True)
X = df[['Age', 'Speed']]  
y = df['Survived']  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)
print("\nModel Accuracy:", accuracy)
print("\nConfusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(class_report)
fig, ax = plt.subplots(figsize=(6, 6))
ax.matshow(conf_matrix, cmap='Blues', alpha=0.7)
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        ax.text(j, i, str(conf_matrix[i, j]), ha='center', va='center', color='black')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
age = 45
speed = 60
new_data = pd.DataFrame([[age, speed]], columns=['Age', 'Speed'])  
survival_prob = model.predict_proba(new_data)
print(f"\nSurvival probability for a passenger with Age={age} and Speed={speed} mph: {survival_prob[0][1]:.2f}")
