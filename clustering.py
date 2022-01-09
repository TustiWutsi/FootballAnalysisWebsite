import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import time
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from FIFA_datasets import db_21

from kmeans_interp.kmeans_feature_imp import KMeansInterp

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

### TEAMS CLUSTERING ###
game_characteristics = ['club_name',
       'potential',
       'skill_moves', 'pace', 'shooting',
       'passing', 'dribbling', 'defending', 'physic', 'attacking_crossing', 'attacking_finishing',
       'attacking_heading_accuracy', 'attacking_short_passing',
       'attacking_volleys', 'skill_dribbling', 'skill_curve',
       'skill_fk_accuracy', 'skill_long_passing', 'skill_ball_control',
       'movement_acceleration', 'movement_sprint_speed', 'movement_agility',
       'movement_reactions', 'movement_balance', 'power_shot_power',
       'power_jumping', 'power_stamina', 'power_strength', 'power_long_shots',
       'mentality_aggression', 'mentality_interceptions',
       'mentality_positioning', 'mentality_vision', 'mentality_penalties',
       'mentality_composure', 'defending_standing_tackle',
       'defending_sliding_tackle']

db_21_clubs = db_21[game_characteristics].dropna().groupby('club_name').mean().reset_index()
db_21_clubs_num = db_21_clubs.drop(columns=['club_name'])

minmax_scaler = MinMaxScaler()
db_21_clubs_num_scaled = minmax_scaler.fit_transform(db_21_clubs_num)
db_21_clubs_num_minmax_scaled_df = pd.DataFrame(db_21_clubs_num_scaled,columns=db_21_clubs_num.columns)

X = db_21_clubs_num_minmax_scaled_df

kms_clubs = KMeansInterp(
n_clusters=6,
ordered_feature_names=X.columns.tolist(), 
feature_importance_method='wcss_min', # or 'unsup2sup'
).fit(X.values)

labels = kms_clubs.labels_

clubs_with_clusters = pd.concat([db_21_clubs,pd.DataFrame(labels)],axis=1).rename(columns={0:"club_cluster"})

### CLUSTERING V2 ###

db_22 = pd.read_csv('files/players_22.csv')

game_characteristics_22 = [
'pace',
'shooting',
'passing',
'dribbling',
'defending',
'physic',
'attacking_crossing',
'attacking_finishing',
'attacking_heading_accuracy',
'attacking_short_passing',
'attacking_volleys',
'skill_dribbling',
'skill_curve',
'skill_fk_accuracy',
'skill_long_passing',
'skill_ball_control',
'movement_acceleration',
'movement_sprint_speed',
'movement_agility',
'movement_reactions',
'movement_balance',
'power_shot_power',
'power_jumping',
'power_stamina',
'power_strength',
'power_long_shots',
'mentality_aggression',
'mentality_interceptions',
'mentality_positioning',
'mentality_vision',
'mentality_penalties',
'mentality_composure',
'defending_marking_awareness',
'defending_standing_tackle',
'defending_sliding_tackle',
'goalkeeping_diving',
'goalkeeping_handling',
'goalkeeping_kicking',
'goalkeeping_positioning',
'goalkeeping_reflexes']

game_characteristics_22_gk = [
'attacking_crossing',
'attacking_finishing',
'attacking_heading_accuracy',
'attacking_short_passing',
'attacking_volleys',
'skill_dribbling',
'skill_curve',
'skill_fk_accuracy',
'skill_long_passing',
'skill_ball_control',
'movement_acceleration',
'movement_sprint_speed',
'movement_agility',
'movement_reactions',
'movement_balance',
'power_shot_power',
'power_jumping',
'power_stamina',
'power_strength',
'power_long_shots',
'mentality_aggression',
'mentality_interceptions',
'mentality_positioning',
'mentality_vision',
'mentality_penalties',
'mentality_composure',
'defending_marking_awareness',
'defending_standing_tackle',
'defending_sliding_tackle',
'goalkeeping_diving',
'goalkeeping_handling',
'goalkeeping_kicking',
'goalkeeping_positioning',
'goalkeeping_reflexes',
'goalkeeping_speed']

positions_to_delete = ['SUB', 'RES', 'CF', 'LAM', 'RAM', 'LF', 'RF', 'nan']
db_22_vf = db_22[db_22["club_position"].isin(positions_to_delete) == False]

def minmax(df):
    minmax_scaler = MinMaxScaler()
    scaled = minmax_scaler.fit_transform(df)
    df_scaled = pd.DataFrame(scaled,columns=df.columns)
    return df_scaled

def pca(df):
    pca = PCA()
    pca.fit(df)
    transformed = pca.transform(df)
    df_transformed = pd.DataFrame(transformed)
    return df_transformed

def elbow_chart(df):
    inertias = []
    ks = range(1,20)
    for k in ks:
        km_test = KMeans(n_clusters=k).fit(df)
        inertias.append(km_test.inertia_)

    fig = px.line(x=ks, y=inertias)
    fig.update_layout(xaxis_title="Number of clusters", yaxis_title="Inertia")
    st.plotly_chart(fig)

def cluster_viz(nb_cluster, df):
    with st.spinner('Wait for it...'):
        time.sleep(5)
    st.success('Done!')
    kmeans = KMeans(n_clusters = int(nb_cluster), max_iter=20)
    kmeans.fit(df)
    labelling = kmeans.labels_
    fig = px.scatter_3d(df,x=0,y=1,z=2,color=labelling)
    st.plotly_chart(fig)

def interpret_clusters(nb_cluster, nb_features, df1, df2):
    kms = KMeansInterp(
    n_clusters=int(nb_cluster),
    ordered_feature_names=df1.columns.tolist(), 
    feature_importance_method='wcss_min', # or 'unsup2sup'
    ).fit(df1.values)

    fig = make_subplots(rows=1, cols=int(nb_cluster), start_cell="bottom-left")
    for i in range(int(nb_cluster)):
        df_important_features = pd.DataFrame(kms.feature_importances_[i][:nb_features], columns=["Feature", "Weight"])
        fig.add_trace(go.Bar(x=df_important_features.Feature, y=df_important_features.Weight, name=f'cluster n°{i}'), row=1, col=i+1)
    fig.update_layout(height=400, width=800)
    st.plotly_chart(fig)

    labels = pd.DataFrame(kms.labels_).rename(columns={0:"label"})
    df2_vf = df2.reset_index().drop(columns=['index'])
    df_with_label = pd.concat([df2_vf,labels],axis=1)

    #df_important_features = pd.DataFrame(kms.feature_importances_[nb_cluster][:10], columns=["Feature", "Weight"])

    return df_with_label

#def ten_features(df):
