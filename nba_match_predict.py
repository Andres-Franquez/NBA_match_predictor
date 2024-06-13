import pandas as pd

df = pd.read_csv('nba_games.csv', index_col=0)

df = df.sort_values('date')
df = df.reset_index(drop=True)

#delete extra columns
del df['mp.1']
del df['mp_opp.1']
del df['index_opp']

#add target frame
def add_target(team):
    team['target'] = team['won'].shift(-1)
    return team

df = df.groupby('team', group_keys=False).apply(add_target)

df['target'][pd.isnull(df['target'])] = 2

#transform t and f in the target col
df['target'] = df['target'].astype(int, errors='ignore')

#remove additional columns -null values-
nulls = pd.isnull(df)
nulls = nulls.sum()
nulls = nulls[nulls > 0]
#nulls

valid_columns = df.columns[~df.columns.isin(nulls.index)]
#valid_columns

df = df[valid_columns].copy()

from sklearn.model_selection import TimeSeriesSplit
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import RidgeClassifier

rr = RidgeClassifier(alpha=1)
split = TimeSeriesSplit(n_splits=3)

sfs = SequentialFeatureSelector(rr, n_features_to_select=30, direction='forward', cv=split)

removed_columns = ['season', 'date', 'won', 'target', 'team', 'team_opp']

selected_columns = df.columns[~df.columns.isin(removed_columns)]

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
df[selected_columns] = scaler.fit_transform(df[selected_columns])

#target columns have been scaled to fall between 0 and 1
#print(df)

sfs.fit(df[selected_columns], df['target'])
#Output: SequentialFeatureSelector(cv=TimeSeriesSplit(gap=0, max_train_size=None, n_splits=3, test_size=None),estimator=RidgeClassifier(alpha=1),n_features_to_select=30)

#get the list of predictors out of seq feat select
#if feature select thinks the column should be used will show True
predictors = list(selected_columns[sfs.get_support()])

def backtest(data, model ,predictors, start=2, step=1):
    all_predictions = []
    
    seasons = sorted(data['season'].unique())
    
    for i in range(start, len(seasons), step):
        season = seasons[i]
        
        train = data[data['season'] < season]
        test = data[data['season'] == season]
        
        model.fit(train[predictors], train['target'])
        
        preds = model.predict(test[predictors])
        preds = pd.Series(preds, index=test.index)
        
        combined = pd.concat([test['target'], preds], axis=1)
        combined.columns = ['actual', 'prediction']
        
        all_predictions.append(combined)
    return pd.concat(all_predictions)


#written function can now work machine learning
predictions = backtest(df, rr, predictors)

#can see actual and the prediction
predictions
#Output
actual	prediction
5250	1	1
5251	1	1
5252	0	0
5253	1	0
5254	0	1
...	...	...
17767	0	0
17768	1	1
17769	0	1
17770	2	1
17771	2	1
12522 rows × 2 columns

#create a metric to see how accurate the model was 
#can see prediction was correct 54.7% of the time
from sklearn.metrics import accuracy_score

accuracy_score(predictions['actual'], predictions['prediction'])
#Output: 0.547196933397220

#to improve model, set baseline for what is good accuracy, home advantage, away disadvantage
df.groupby('home').apply(lambda x: x[x['won'] == 1].shape[0] / x.shape[0])

df_rolling = df[list(selected_columns) + ['won', 'team', 'season']]

#find team average of performance in last 10 games, use rolling method 
def find_team_averages(team):
    rolling = team.rolling(10).mean()
    return rolling

df_rolling = df_rolling.groupby(['team', 'season'], group_keys=False).apply(find_team_averages)

#combine rolling columns with regular columns 
rolling_cols = [f'{col}_10' for col in df_rolling.columns]
df_rolling.columns = rolling_cols

df = pd.concat([df, df_rolling], axis=1)

#drop with missing values in the dataset
df = df.dropna()

#give the algorithim more information for better accurracy 
def shift_col(team, col_name):
    next_col = team[col_name].shift(-1)
    return next_col

def add_col(df, col_name): # pass in team data 
    return df.groupby('team', group_keys=False).apply(lambda x: shift_col(x, col_name))

df['home_next'] = add_col(df, 'home')
df['team_opp_next'] = add_col(df, 'team_opp')
df['date_next'] = add_col(df, 'date')

#pull in stats about opp teams
full = df.merge(
    df[rolling_cols + ['team_opp_next', 'date_next', 'team']], 
    left_on=['team', 'date_next'],
    right_on=['team_opp_next', 'date_next']
    )

#look at a few column to look better at the data after merge
full[['team_x', 'team_opp_next_x', 'team_y', 'team_opp_next_y', 'date_next']]
#Output:
	team_x	team_opp_next_x	team_y	team_opp_next_y	date_next
0	SAC	TOR	TOR	SAC	2015-11-15
1	TOR	SAC	SAC	TOR	2015-11-15
2	CLE	DET	DET	CLE	2015-11-17
3	GSW	TOR	TOR	GSW	2015-11-17
4	DEN	NOP	NOP	DEN	2015-11-17
...	...	...	...	...	...
15769	BOS	GSW	GSW	BOS	2022-06-10
15770	GSW	BOS	BOS	GSW	2022-06-13
15771	BOS	GSW	GSW	BOS	2022-06-13
15772	GSW	BOS	BOS	GSW	2022-06-16
15773	BOS	GSW	GSW	BOS	2022-06-16
15774 rows × 5 columns

#run through seq feat to see which is best 
#create set of columns to not run through model
removed_columns = list(full.columns[full.dtypes == 'object']) + removed_columns

selected_columns = full.columns[~full.columns.isin(removed_columns)] 

sfs.fit(full[selected_columns], full['target'])

predictors = list(selected_columns[sfs.get_support()])
predictions = backtest(full, rr, predictors)
#define accuracy score to see improvement of 62.9%
accuracy_score(predictions['actual'], predictions['prediction'])
#Output: 0.629629629629629












