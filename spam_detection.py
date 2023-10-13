import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import sys
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Define the pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)),
    ('model', LogisticRegression())
])

# Load the saved model
pipeline = joblib.load('spam_model.pkl')

# Example usage
if len(sys.argv) > 1:
    input_mail = [sys.argv[1]]
    prediction = pipeline.predict(input_mail)

    if prediction == 1:
        print("This is a Ham Mail.")
    else:
        print("This is a Spam Mail.")
else:
    print("Please provide text input.")