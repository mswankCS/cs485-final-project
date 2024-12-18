import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Name of column in dataset we intend to target
target_column = 'voted_up'
drop_columns = ['recommendationid', 'appid', 'game', 'author_steamid', 'author_num_games_owned', 'author_num_reviews', 
                'author_playtime_forever', 'author_playtime_last_two_weeks', 'author_playtime_at_review', 
                'author_last_played', 'language', 'timestamp_created', 'timestamp_updated', 'votes_up',
                'votes_funny', 'weighted_vote_score', 'comment_count', 'steam_purchase', 'received_for_free', 
                'written_during_early_access', 'hidden_in_steam_china', 'steam_china_location'] #
                # kept: review, voted_up

"""
With these settings, the Logistic Regression model will only use the 'review' column (reviews represented as BoW)
to predict 'voted up' value (either 0 or 1, 0 being neg and 1 being pos). All other columns listed in drop_columns
are disregarded. It also only uses reviews marked as 'english'. Dataset being used is 'weighted_score_above_08.csv',
not the 42GB 'all_reviews.csv'. Solver being used is 'lbfgs' (Haven't experimented with the others yet)
"""

# Load dataset
"""Dataset not added to github.
The file structure I have on my machine is ./data/weighted_score_above_08.csv and ./data/all_reviews/all_reviews.csv"""
data = pd.read_csv('data/weighted_score_above_08.csv', low_memory=False)

# Keep only english reviews
data = data[data['language'] == 'english']

# append playtime at the end as an additional 'word'
# data = data.astype({'author_playtime_at_review': '<U11'})

def categorize_playtime(row):
    # if row['author_playtime_at_review'] < 100:
    #     return ' playtimeUnder100'
    # elif row['author_playtime_at_review'] < 1000:
    #     return ' playtimeUnder1000'
    # elif row['author_playtime_at_review'] < 10000:
    #     return ' playtimeUnder10000'
    # elif row['author_playtime_at_review'] < 100000:
    #     return ' playtimeUnder100000'
    # else:
    #     return ' playtimeOver100000'
    if row['author_playtime_at_review'] < 1000:
        return ' playtimeUnder1000'
    else:
        return ' playtimeOver1000'

data['review'] = data['review'] + data.apply(categorize_playtime, axis=1)

# Drop unwanted columns
for column_name in drop_columns:
    if column_name in data.columns:
        data = data.drop(columns=column_name)

# Make sure it looks right
print(data.head())
# print(data.info())
# print(data.columns)

# Convert reviews into BoW representations
vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(data['review'])

# Train/test split
y = data[target_column]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

# Set up and train Logistic Regression model
model = LogisticRegression(max_iter=1000, solver='lbfgs')
model.fit(X_train, y_train)

# Make predictions using model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print('Accuracy: ', accuracy)
print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))