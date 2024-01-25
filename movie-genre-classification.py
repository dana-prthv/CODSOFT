import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load train and test data
train_data = pd.read_csv('C:\\Users\\world\\OneDrive\\Desktop\\INTERNSHIPS\\Genre Classification Dataset\\train_data.txt', delimiter=' ::: ', engine='python', header=None, names=['ID', 'TITLE', 'GENRE', 'DESCRIPTION'])
test_data = pd.read_csv('C:\\Users\\world\\OneDrive\\Desktop\\INTERNSHIPS\\Genre Classification Dataset\\test_data.txt', delimiter=' ::: ', engine='python', header=None, names=['ID', 'TITLE', 'DESCRIPTION'])
test_data_solution = pd.read_csv('C:\\Users\\world\\OneDrive\\Desktop\\INTERNSHIPS\\Genre Classification Dataset\\test_data_solution.txt', delimiter=' ::: ', engine='python', header=None, names=['ID', 'TITLE', 'GENRE', 'DESCRIPTION'])

print(train_data.head(25))

# Combine train and test data solution for genre distribution
combined_data = pd.concat([train_data, test_data_solution])

# Plot bar graph for the number of movies in each genre
plt.figure(figsize=(12, 6))
sns.countplot(x='GENRE', data=combined_data, order=combined_data['GENRE'].value_counts().index)
plt.xlabel('Genre')
plt.ylabel('Number of Movies')
plt.title('Number of Movies in Each Genre')
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
plt.show()



# Split the train data into training and validation sets
train_set, val_set = train_test_split(train_data, test_size=0.2, random_state=42)

# Preprocess text data using bag-of-words
vectorizer = CountVectorizer(stop_words='english')
X_train = vectorizer.fit_transform(train_set['DESCRIPTION'])
y_train = train_set['GENRE']

# Train a Naive Bayes classifier
classifier = MultinomialNB()
classifier.fit(X_train, y_train)

# Predict on the validation set
X_val = vectorizer.transform(val_set['DESCRIPTION'])
y_val_pred = classifier.predict(X_val)

print(test_data.head(25))


# Evaluate the model
accuracy = accuracy_score(val_set['GENRE'], y_val_pred)
print(f'Validation Accuracy: {accuracy:.2f}')

print(test_data_solution.head(25))

# Predict on the test set
X_test = vectorizer.transform(test_data['DESCRIPTION'])
y_test_pred = classifier.predict(X_test)

# Compare predictions with the test data solution
y_test_true = test_data_solution['GENRE']

# Create a confusion matrix
conf_matrix = confusion_matrix(y_test_true, y_test_pred)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=classifier.classes_, yticklabels=classifier.classes_)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()


