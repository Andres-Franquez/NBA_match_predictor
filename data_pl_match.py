import pandas as pd
matches = pd.read_csv('matches.csv', index_col=0)
#Take a look at the data
matches.head() 

#investigate missing data
matches['team'].value_counts()

#Can see that 7 teams have less than 70 matches
#Liverpool was not relagated in the year 2022
matches[matches['team'] == 'Liverpool']

matches['round'].value_counts()

#Clean the data for machine learning
#machine learning algorithims can only work with numeric values, not objects
matches.dtypes 
#convert date col, into date time
matches['date'] = pd.to_datetime(matches['date'])
matches.dtypes

#creating predictors for machine learning
matches['venue_code'] = matches['venue'].astype('category').cat.codes

matches['opp_code'] = matches['opponent'].astype('category').cat.codes

matches['hour'] = matches['time'].str.replace(':.+', "", regex=True).astype('int')

matches['day_code'] = matches['date'].dt.dayofweek

matches['target'] = (matches['result'] == "W").astype("int")

#creating the inital machine learning model
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=50, min_samples_split=10, random_state=1)

train = matches[matches['date'] < '2022-01-01']
test = matches[matches['date'] > '2022-01-01']
predictors = ['venue_code', 'opp_code', 'hour', 'day_code']

rf.fit(train[predictors], train['target'])
preds= rf.predict(test[predictors])


from sklearn.metrics import accuracy_score
acc = accuracy_score(test['target'], preds)
#Output: 0.6123188405797102

combine = pd.DataFrame(dict(actual= test['target'], prediction = preds))

#Found when predicted draw or loss it was more likely to be right
#Found when predicted a win, it was more likely to be wrong 
pd.crosstab(index = combine['actual'], columns=combine['prediction'])

#only care for predicting wins 
#use precision_score to be more accurate
from sklearn.metrics import precision_score

precision_score(test['target'], preds)
#Output: 0.4745762711864407












