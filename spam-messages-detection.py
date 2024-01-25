# Importing necessary libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Loading the SMS spam dataset (replace 'path/to/spam_dataset.csv' with your actual file path)
data = pd.read_csv('C:\\Users\\world\\OneDrive\\Desktop\\INTERNSHIPS\\spam.csv', encoding='latin-1')
print(data)
data = data[['v1', 'v2']]  # Selecting only the relevant columns
data.columns = ['label', 'message']  # Rename columns for better readability

# Converting labels to binary (spam: 1, ham: 0)
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data['message'], data['label'], test_size=0.2, random_state=42)

# Creating a TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer()

# Transforming the training data
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)

# Transforming the test data
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Creating a logistic regression model
model = LogisticRegression()

# Training the model
model.fit(X_train_tfidf, y_train)

# Making predictions on the test set
predictions = model.predict(X_test_tfidf)

# Evaluating the model
accuracy = accuracy_score(y_test, predictions)
conf_matrix = confusion_matrix(y_test, predictions)
classification_rep = classification_report(y_test, predictions)

print(f"Accuracy: {accuracy}")
print("\n\n\n\n")
print(f"Classification Report:\n{classification_rep}")
print("\n\n\n\n")



# Printing first 40 spam messages
spam_messages = data[data['label'] == 1]['message'].tolist()
print("\nFirst 40 Spam Messages:")
for msg in spam_messages[:40]:  # Print the first 25 spam messages
    print(f"- {msg}")

# Visualizing results using confusion matrix
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()


# Visualizing results using histograms and box plots
plt.figure(figsize=(12, 6))

# Histogram of true labels
plt.subplot(1, 2, 1)
sns.histplot(data['label'], bins=[-0.5, 0.5, 1.5], kde=False, color='skyblue', edgecolor='black')
plt.xticks([0, 1], ['Ham', 'Spam'])
plt.title('True Label Distribution')

# Histogram of predicted labels
plt.subplot(1, 2, 2)
sns.histplot(predictions, bins=[-0.5, 0.5, 1.5], kde=False, color='lightcoral', edgecolor='black')
plt.xticks([0, 1], ['Ham', 'Spam'])
plt.title('Predicted Label Distribution')

plt.tight_layout()
plt.show()
