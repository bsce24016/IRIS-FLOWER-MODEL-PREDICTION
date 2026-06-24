# ============================
# Iris Flower Classification
# ============================

# 1. Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 2. Load dataset
df = pd.read_csv("Iris.csv")   # change filename if your file name is different

# 3. Display first 5 rows
print("First 5 rows of dataset:")
print(df.head())

# 4. Check dataset info
print("\nDataset Info:")
print(df.info())

# 5. Check for missing values
print("\nMissing values in each column:")
print(df.isnull().sum())

# 6. Drop Id column if it exists
if 'Id' in df.columns:
    df = df.drop('Id', axis=1)

# 7. Separate features and target
X = df.drop('species', axis=1)   # input features
y = df['species']                # target column

# 8. Encode target labels into numbers
# Example:
# Iris-setosa -> 0
# Iris-versicolor -> 1
# Iris-virginica -> 2
le = LabelEncoder()
y_encoded = le.fit_transform(y)

print("\nEncoded classes:")
for i, class_name in enumerate(le.classes_):
    print(f"{class_name} -> {i}")

# 9. Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

print("\nTraining data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)

# 10. Create KNN model
model = KNeighborsClassifier(n_neighbors=3)

# 11. Train the model
model.fit(X_train, y_train)

# 12. Make predictions on test data
y_pred = model.predict(X_test)

# 13. Evaluate model accuracy
accuracy = accuracy_score(y_test, y_pred)
print("\nModel Accuracy:", accuracy * 100, "%")

# 14. Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# 15. Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# 16. Plot confusion matrix
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, cmap="Blues", fmt="d",
            xticklabels=le.classes_,
            yticklabels=le.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Iris Classification")
plt.show()