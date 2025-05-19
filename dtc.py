import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils import resample
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("tweets.csv")  
df = df[['text', 'target']]

# Balance dataset
df_disaster = df[df['target'] == 1]
df_non_disaster = df[df['target'] == 0]
df_non_disaster_downsampled = resample(df_non_disaster, replace=False, n_samples=len(df_disaster), random_state=42)
df_balanced = pd.concat([df_non_disaster_downsampled, df_disaster])
df_balanced = df_balanced.sample(frac=1, random_state=42)  # Shuffle

# Text preprocessing
nltk.download('stopwords')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
important_words = {"earthquake", "hurricane", "tremor", "tsunami", "flood", "wildfire"}

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(f"[{string.punctuation}]", "", text)  # Remove punctuation
    words = text.split()
    words = [word for word in words if word in important_words or word not in stop_words]
    words = [lemmatizer.lemmatize(word) for word in words]  # Lemmatization
    return " ".join(words)

df_balanced['text'] = df_balanced['text'].apply(preprocess_text)

# Splitting data
X_train, X_test, y_train, y_test = train_test_split(df_balanced['text'], df_balanced['target'], test_size=0.2, random_state=42)

# Vectorization
vectorizer = TfidfVectorizer(ngram_range=(1,1), max_features=5000)  # Using only unigrams, limiting features
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Model training
model = LogisticRegression(max_iter=500, solver='lbfgs')  # Faster model
model.fit(X_train_vec, y_train)

# Predictions
y_pred = model.predict(X_test_vec)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.4f}')
print(classification_report(y_test, y_pred))

# Function to predict new tweets
def classify_tweet(tweet):
    processed_tweet = preprocess_text(tweet)
    tweet_vec = vectorizer.transform([processed_tweet])
    prediction = model.predict(tweet_vec)[0]
    return "Disaster Tweet" if prediction == 1 else "Not a Disaster Tweet"

# Example usage
while True:
    user_input = input("Enter a tweet (or type 'exit' to quit): ")
    if user_input.lower() == 'exit':
        break
    print("Classification:", classify_tweet(user_input))
