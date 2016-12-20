import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble                import RandomForestClassifier
from sklearn.cross_validation        import train_test_split
from sklearn.metrics                 import log_loss

def add_tag_dummies(df, tags):
    tag_cols = df.loc[:, tag_columns]

    for tag in tags:
        tag_present = pd.concat([tag_cols[col] == tag for col in tag_columns], axis = 1).any(axis = 1)
        df['Tag ' + tag] = tag_present

    df.drop(tag_columns, axis = 1, inplace = True)

def extract_information(train_df, test_df, column_name, max_features):
    vectorizer = CountVectorizer(max_features=max_features)
    lowercase_train_col = train_df[column_name].apply(str.lower)    
    lowercase_test_col  = test_df[column_name].apply(str.lower)
    # Count the words in train and only keep the 'max_features' most common
    vectorizer.fit(lowercase_train_col)
    # Associate to each entry a (1, max_features) vector corresponding to 
    # the number of occurences of a given word
    train_bow_matrix = vectorizer.transform(lowercase_train_col).todense()
    test_bow_matrix  = vectorizer.transform(lowercase_test_col ).todense()
    # Combine all these results in a pandas dataframe with proper columns names
    column_names = ['%s %s' % (column_name, word) for word in vectorizer.get_feature_names()]
    
    train_df_expansion = pd.DataFrame(train_bow_matrix, index = train_df.index, columns = column_names)
    test_df_expansion  = pd.DataFrame(test_bow_matrix , index = test_df.index , columns = column_names)

    expanded_train_df = train_df.drop([column_name], axis = 1).join(train_df_expansion)
    expanded_test_df  = test_df.drop( [column_name], axis = 1).join(test_df_expansion )
    
    return expanded_train_df, expanded_test_df

train = pd.read_csv('../input/train.csv')[:5000]
leaderboard = pd.read_csv('../input/public_leaderboard.csv')[:1000]

train.drop(['PostId', 'PostCreationDate', 'OwnerUserId', 'OwnerCreationDate', 'PostClosedDate'], axis = 1, inplace = True)
leaderboard.drop(['PostCreationDate', 'OwnerUserId', 'OwnerCreationDate'], axis = 1, inplace = True)

tag_columns = ['Tag%d' % tag_number for tag_number in range(1, 6)]
tags = train.loc[:, tag_columns]
all_tags = pd.concat([tags[col] for col in tags.columns], axis = 0).dropna()

tag_frequency = all_tags.value_counts()
most_common_tags = tag_frequency[tag_frequency > 500]
    
tags = most_common_tags.keys()
add_tag_dummies(train, tags)
add_tag_dummies(leaderboard, tags)

train, leaderboard = extract_information(train, leaderboard, 'Title'       , 200)
train, leaderboard = extract_information(train, leaderboard, 'BodyMarkdown', 500)

X_df = train.drop(['Unnamed: 0', 'OpenStatus'], axis = 1)
y_df = train['OpenStatus']
X    = X_df.as_matrix()
y    = y_df.as_matrix()

X_leaderboard_df = leaderboard.drop(['PostId'], axis = 1)
X_leaderboard = X_leaderboard_df.as_matrix()

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y)

clf = RandomForestClassifier(n_estimators=100, random_state=142857)
clf.fit(Xtrain, ytrain)

leaderboard_predictions = clf.predict_proba(X_leaderboard)[:, 1]
submission_dict = {'id'        : leaderboard['PostId'],
                   'OpenStatus': leaderboard_predictions}
submission_df = pd.DataFrame(submission_dict, index=leaderboard.index)

submission_df.to_csv('../submission/submission.csv')
