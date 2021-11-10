import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from FIFA_datasets import db_21

### PLAYERS CLUSTERING ###

#Spliting players according to position
strikers = db_21[db_21['position'] == 'striker']

#Keeping only numeric and useful columns for clustering
strikers_num = strikers.select_dtypes(exclude='object').drop(columns=['sofifa_id', 'team_jersey_number', 'nation_jersey_number', 'gk_diving', 'gk_handling', 'gk_kicking', 'gk_reflexes', 'gk_speed', 'gk_positioning', 'defending_marking']).dropna(subset=['release_clause_eur'])

#Applying clustering 
#STRIKERS
minmax_scaler = MinMaxScaler()
strikers_num_scaled = minmax_scaler.fit_transform(strikers_num)
strikers_num_minmax_scaled_df = pd.DataFrame(strikers_num_scaled,columns=strikers_num.columns)

pca = PCA()
pca.fit(strikers_num_minmax_scaled_df)

strikers_num_scaled_and_transformed = pca.transform(strikers_num_minmax_scaled_df)
strikers_num_scaled_and_transformed = pd.DataFrame(strikers_num_scaled_and_transformed)

kmeans = KMeans(n_clusters = 6, max_iter=20)
kmeans.fit(strikers_num_scaled_and_transformed)
labelling = kmeans.labels_
