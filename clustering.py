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

strikers_with_label = pd.concat([strikers,pd.Series(labelling)],axis=1).rename(columns={0:"player_cluster"}).dropna(subset=['short_name','player_cluster'])

def strikers_details(cluster_number):
    if cluster_number == 0:
        return 'Quite technical / fast / Correct finisher / powerful physic / Aggressive'
    if cluster_number == 1:
        return 'Experiment strikers / low skills and low speed / Tall, heavy and powerful'
    if cluster_number == 2:
        return 'Top class striker / Very high skills / Finisher / High potential / Small but still quite powerful and fast'
    if cluster_number == 3:
        return 'Tall and powerful strikers / Young and speed / aggressive / low skills but good heading'
    if cluster_number == 4:
        return 'Very Strenghful / Good positionning / Correct technic but not fast / Correct finisher'
    if cluster_number == 5:
        return 'High skills / Potential / lack of strengh and physic / low heading and jump'

strikers_with_label['cluster_description'] = strikers_with_label['player_cluster'].map(strikers_details)

strikers_clusters = pd.read_csv('files/strikers_clusters.csv')