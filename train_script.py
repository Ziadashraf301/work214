import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
from sklearn.pipeline import Pipeline
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Load Data from csv file to a pandas datafram
raw_mail_data = pd.read_csv('./mail_data.csv')

# Replace the null values with a null string
mail_data = raw_mail_data.where(pd.notnull(raw_mail_data),'')

#  Label Encoding
mail_data.loc[mail_data['Category'] == 'spam', 'Category'] = 0
mail_data.loc[mail_data['Category'] == 'ham', 'Category'] = 1


# Seperating the text as texts and label
X = mail_data['Message']
Y = mail_data['Category'].astype('int')


# Define the pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)),
    ('model', LogisticRegression())
])

# Train the model
pipeline.fit(X, Y)


# Save the trained model
joblib.dump(pipeline, 'spam_model.pkl')
print("The model was saved.")
