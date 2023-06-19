import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import BernoulliNB

# Load the data
df = pd.read_csv("C:/Users/pc work/Desktop/Project/Dataset.csv")
df.drop(df.columns[[0, 1, 2, 7, 10, 11, 12, 13, 14]], axis=1, inplace=True)

# Select the categorical columns
cat_cols = df.select_dtypes(include=["object"]).columns

# Encode the categorical columns
le = LabelEncoder()
df[cat_cols] = df[cat_cols].apply(lambda col: le.fit_transform(col))

# Select the input features and the target variable
X = df.drop("hospital_death", axis=1)
y = df["hospital_death"]

# Create an imputer
imputer = SimpleImputer(strategy="mean")
X = imputer.fit_transform(X)

# Split the data into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create the KNN classifier
clf = BernoulliNB()

# Train the classifier on the training set
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
