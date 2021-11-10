from sklearn.preprocessing import MinMaxScaler

from FIFA_datasets import db_21

### PLAYERS CLUSTERING ###

#Spliting players according to position
strikers = db_21[db_21['position'] == 'striker']

#Keeping only numeric and useful columns for clustering
strikers_num = strikers.select_dtypes(exclude='object').drop(columns=['sofifa_id', 'team_jersey_number', 'nation_jersey_number', 'gk_diving', 'gk_handling', 'gk_kicking', 'gk_reflexes', 'gk_speed', 'gk_positioning', 'defending_marking']).dropna(subset=['release_clause_eur'])

