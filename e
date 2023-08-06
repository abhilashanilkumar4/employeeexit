import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

data_url = 'https://github.com/edyoda/data-science-complete-tutorial/blob/master/Data/HR_comma_sep.csv.txt'
df = pd.read_csv(data_url)

print(df.head())
print(df.info())

X = df.drop('left', axis=1)
y = df['left']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)

print("Accuracy:", accuracy)
print("Confusion Matrix:")
print(confusion)

sns.countplot(x='left', data=df)
plt.title('Distribution of Employee Exits')
plt.xlabel('Exit (1) / Not Exit (0)')
plt.ylabel('Count')
plt.show()
