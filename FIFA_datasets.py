import pandas as pd
from functools import reduce

def convert_position(position):
    if position in ['RB', 'RWB', 'LB', 'LWB','CB', 'RCB', 'LCB']:
        return 'defender'
    if position in ['CDM','CM','CAM','LM','RM','LDM','RDM','LCM','RDM']:
        return 'midfielder'
    if position in ['LW','LF','RW','RF','CF','ST','LS','RS','LAM','RAM']:
        return 'striker'
    if position in ['GK']:
        return 'goalkeeper'
    if position in ['SUB']:
        return 'substitute'
    if position in ['RES']:
        return 'reserve'
    else:
        return 'nan'


#2015
db_15 = pd.read_csv('files/players_15.csv')

db_15['position'] = db_15['team_position'].map(convert_position)

position_ranking_15 = db_15.groupby(['position','league_name'])[['overall']].mean().reset_index()

defenders_15 = position_ranking_15[position_ranking_15['position'] == 'defender'].rename(columns = {'overall': 'def_15'}).drop(columns = ['position'])
midfielders_15 = position_ranking_15[position_ranking_15['position'] == 'midfielder'].rename(columns = {'overall': 'mid_15'}).drop(columns = ['position'])
strikers_15 = position_ranking_15[position_ranking_15['position'] == 'striker'].rename(columns = {'overall': 'str_15'}).drop(columns = ['position'])
goalkeeper_15 = position_ranking_15[position_ranking_15['position'] == 'goalkeeper'].rename(columns = {'overall': 'gk_15'}).drop(columns = ['position'])
substitutes_15 = position_ranking_15[position_ranking_15['position'] == 'substitute'].rename(columns = {'overall': 'sub_15'}).drop(columns = ['position'])
overall_15 = db_15.groupby(['league_name'])[['overall']].mean().rename(columns = {'overall': 'overall_15'}).reset_index()

df_15 = [overall_15, defenders_15, midfielders_15, strikers_15, goalkeeper_15, substitutes_15]

position_15 = reduce(lambda  left,right: pd.merge(left,right,on=['league_name'],
                                            how='outer'), df_15)

#2016
db_16 = pd.read_csv('files/players_16.csv')

db_16['position'] = db_16['team_position'].map(convert_position)

position_ranking_16 = db_16.groupby(['position','league_name'])[['overall']].mean().reset_index()

defenders_16 = position_ranking_16[position_ranking_16['position'] == 'defender'].rename(columns = {'overall': 'def_16'}).drop(columns = ['position'])
midfielders_16 = position_ranking_16[position_ranking_16['position'] == 'midfielder'].rename(columns = {'overall': 'mid_16'}).drop(columns = ['position'])
strikers_16 = position_ranking_16[position_ranking_16['position'] == 'striker'].rename(columns = {'overall': 'str_16'}).drop(columns = ['position'])
goalkeeper_16 = position_ranking_16[position_ranking_16['position'] == 'goalkeeper'].rename(columns = {'overall': 'gk_16'}).drop(columns = ['position'])
substitutes_16 = position_ranking_16[position_ranking_16['position'] == 'substitute'].rename(columns = {'overall': 'sub_16'}).drop(columns = ['position'])
overall_16 = db_16.groupby(['league_name'])[['overall']].mean().rename(columns = {'overall': 'overall_16'}).reset_index()

df_16 = [overall_16, defenders_16, midfielders_16, strikers_16, goalkeeper_16, substitutes_16]

position_16 = reduce(lambda  left,right: pd.merge(left,right,on=['league_name'],
                                            how='outer'), df_16)

#2017
db_17 = pd.read_csv('files/players_17.csv')

db_17['position'] = db_17['team_position'].map(convert_position)

position_ranking_17 = db_17.groupby(['position','league_name'])[['overall']].mean().reset_index()

defenders_17 = position_ranking_17[position_ranking_17['position'] == 'defender'].rename(columns = {'overall': 'def_17'}).drop(columns = ['position'])
midfielders_17 = position_ranking_17[position_ranking_17['position'] == 'midfielder'].rename(columns = {'overall': 'mid_17'}).drop(columns = ['position'])
strikers_17 = position_ranking_17[position_ranking_17['position'] == 'striker'].rename(columns = {'overall': 'str_17'}).drop(columns = ['position'])
goalkeeper_17 = position_ranking_17[position_ranking_17['position'] == 'goalkeeper'].rename(columns = {'overall': 'gk_17'}).drop(columns = ['position'])
substitutes_17 = position_ranking_17[position_ranking_17['position'] == 'substitute'].rename(columns = {'overall': 'sub_17'}).drop(columns = ['position'])
overall_17 = db_17.groupby(['league_name'])[['overall']].mean().rename(columns = {'overall': 'overall_17'}).reset_index()

df_17 = [overall_17, defenders_17, midfielders_17, strikers_17, goalkeeper_17, substitutes_17]

position_17 = reduce(lambda  left,right: pd.merge(left,right,on=['league_name'],
                                            how='outer'), df_17)

#2018
db_18 = pd.read_csv('files/players_18.csv')

db_18['position'] = db_18['team_position'].map(convert_position)

position_ranking_18 = db_18.groupby(['position','league_name'])[['overall']].mean().reset_index()

defenders_18 = position_ranking_18[position_ranking_18['position'] == 'defender'].rename(columns = {'overall': 'def_18'}).drop(columns = ['position'])
midfielders_18 = position_ranking_18[position_ranking_18['position'] == 'midfielder'].rename(columns = {'overall': 'mid_18'}).drop(columns = ['position'])
strikers_18 = position_ranking_18[position_ranking_18['position'] == 'striker'].rename(columns = {'overall': 'str_18'}).drop(columns = ['position'])
goalkeeper_18 = position_ranking_18[position_ranking_18['position'] == 'goalkeeper'].rename(columns = {'overall': 'gk_18'}).drop(columns = ['position'])
substitutes_18 = position_ranking_18[position_ranking_18['position'] == 'substitute'].rename(columns = {'overall': 'sub_18'}).drop(columns = ['position'])
overall_18 = db_18.groupby(['league_name'])[['overall']].mean().rename(columns = {'overall': 'overall_18'}).reset_index()

df_18 = [overall_18, defenders_18, midfielders_18, strikers_18, goalkeeper_18, substitutes_18]

position_18 = reduce(lambda  left,right: pd.merge(left,right,on=['league_name'],
                                            how='outer'), df_18)

#2019
db_19 = pd.read_csv('files/players_19.csv')

db_19['position'] = db_19['team_position'].map(convert_position)

position_ranking_19 = db_19.groupby(['position','league_name'])[['overall']].mean().reset_index()

defenders_19 = position_ranking_19[position_ranking_19['position'] == 'defender'].rename(columns = {'overall': 'def_19'}).drop(columns = ['position'])
midfielders_19 = position_ranking_19[position_ranking_19['position'] == 'midfielder'].rename(columns = {'overall': 'mid_19'}).drop(columns = ['position'])
strikers_19 = position_ranking_19[position_ranking_19['position'] == 'striker'].rename(columns = {'overall': 'str_19'}).drop(columns = ['position'])
goalkeeper_19 = position_ranking_19[position_ranking_19['position'] == 'goalkeeper'].rename(columns = {'overall': 'gk_19'}).drop(columns = ['position'])
substitutes_19 = position_ranking_19[position_ranking_19['position'] == 'substitute'].rename(columns = {'overall': 'sub_19'}).drop(columns = ['position'])
overall_19 = db_19.groupby(['league_name'])[['overall']].mean().rename(columns = {'overall': 'overall_19'}).reset_index()

df_19 = [overall_19, defenders_19, midfielders_19, strikers_19, goalkeeper_19, substitutes_19]

position_19 = reduce(lambda  left,right: pd.merge(left,right,on=['league_name'],
                                            how='outer'), df_19)

#2020
db_20 = pd.read_csv('files/players_20.csv')

db_20['position'] = db_20['team_position'].map(convert_position)

position_ranking_20 = db_20.groupby(['position','league_name'])[['overall']].mean().reset_index()

defenders_20 = position_ranking_20[position_ranking_20['position'] == 'defender'].rename(columns = {'overall': 'def_20'}).drop(columns = ['position'])
midfielders_20 = position_ranking_20[position_ranking_20['position'] == 'midfielder'].rename(columns = {'overall': 'mid_20'}).drop(columns = ['position'])
strikers_20 = position_ranking_20[position_ranking_20['position'] == 'striker'].rename(columns = {'overall': 'str_20'}).drop(columns = ['position'])
goalkeeper_20 = position_ranking_20[position_ranking_20['position'] == 'goalkeeper'].rename(columns = {'overall': 'gk_20'}).drop(columns = ['position'])
substitutes_20 = position_ranking_20[position_ranking_20['position'] == 'substitute'].rename(columns = {'overall': 'sub_20'}).drop(columns = ['position'])
overall_20 = db_20.groupby(['league_name'])[['overall']].mean().rename(columns = {'overall': 'overall_20'}).reset_index()

df_20 = [overall_20, defenders_20, midfielders_20, strikers_20, goalkeeper_20, substitutes_20]

position_20 = reduce(lambda  left,right: pd.merge(left,right,on=['league_name'],
                                            how='outer'), df_20)

#2021
db_21 = pd.read_csv('files/players_21.csv')

db_21['position'] = db_21['team_position'].map(convert_position)

position_ranking_21 = db_21.groupby(['position','league_name'])[['overall']].mean().reset_index()

defenders_21 = position_ranking_21[position_ranking_21['position'] == 'defender'].rename(columns = {'overall': 'def_21'}).drop(columns = ['position'])
midfielders_21 = position_ranking_21[position_ranking_21['position'] == 'midfielder'].rename(columns = {'overall': 'mid_21'}).drop(columns = ['position'])
strikers_21 = position_ranking_21[position_ranking_21['position'] == 'striker'].rename(columns = {'overall': 'str_21'}).drop(columns = ['position'])
goalkeeper_21 = position_ranking_21[position_ranking_21['position'] == 'goalkeeper'].rename(columns = {'overall': 'gk_21'}).drop(columns = ['position'])
substitutes_21 = position_ranking_21[position_ranking_21['position'] == 'substitute'].rename(columns = {'overall': 'sub_21'}).drop(columns = ['position'])
overall_21 = db_21.groupby(['league_name'])[['overall']].mean().rename(columns = {'overall': 'overall_21'}).reset_index()

df_21 = [overall_21, defenders_21, midfielders_21, strikers_21, goalkeeper_21, substitutes_21]

position_21 = reduce(lambda  left,right: pd.merge(left,right,on=['league_name'],
                                            how='outer'), df_21)

#gather datasets
df_all_years = [position_15, position_16, position_17, position_18, position_19, position_20, position_21]

positions_per_champ = reduce(lambda  left,right: pd.merge(left,right,on=['league_name'],
                                            how='outer'), df_all_years)

positions_per_champ['overall_evol'] = positions_per_champ['overall_20'] - positions_per_champ['overall_15']

best_evol_champ = positions_per_champ[positions_per_champ['overall_evol']>2.5].sort_values('overall_evol', ascending=False)

#add latitute and longitude
lat_lon = pd.read_csv('files/lat_lon.csv')
lat_lon = lat_lon[['country','latitude','longitude']]

def league_country(league):
    if league == 'Argentina Primera División':
        return 'Argentina'
    if league == 'Australian Hyundai A-League':
        return 'Australia'
    if league == 'Austrian Football Bundesliga':
        return 'Austria'
    if league == 'Belgian Jupiler Pro League':
        return 'Belgium'
    if league == 'Chilian Campeonato Nacional':
        return 'Chile'
    if league == 'English Premier League':
        return 'United Kingdom'
    if league == 'Spain Primera Division':
        return 'Spain'
    if league == 'French Ligue 1':
        return 'France'
    if league == 'German 1. Bundesliga':
        return 'Germany'
    if league == 'Italian Serie A':
        return 'Italy'
    else:
        return 'nan'

positions_per_champ['country'] = positions_per_champ['league_name'].map(league_country)

map_df = pd.merge(positions_per_champ, lat_lon, on='country')

#Evolution of top 5 european leagues
top_5_list = ['Spain Primera Division', 'English Premier League', 'Italian Serie A', 'German 1. Bundesliga', 'French Ligue 1']
top_5 = positions_per_champ[positions_per_champ['league_name'].isin(top_5_list)][['league_name','overall_15','overall_16','overall_17','overall_18','overall_19','overall_20']]
top_5 = top_5.rename(columns = {'overall_15' : '2015', 'overall_16' : '2016', 'overall_17' : '2017', 'overall_18' : '2018', 'overall_19' : '2019', 'overall_20' : '2020', 'overall_21' : '2021'})

top_5 = top_5.melt(['league_name'], var_name='year').rename(columns = {'value':'league_level'})

#Clubs level and value evolution
club_level_15 = db_15.groupby(['league_name','club_name']).agg({'overall' : 'mean', 'value_eur' : 'sum'}).rename(columns={'overall' : 'overall_15', 'value_eur' : 'value_eur_15'}).reset_index()
club_level_16 = db_16.groupby(['league_name','club_name']).agg({'overall' : 'mean', 'value_eur' : 'sum'}).rename(columns={'overall' : 'overall_16', 'value_eur' : 'value_eur_16'}).reset_index()
club_level_17 = db_17.groupby(['league_name','club_name']).agg({'overall' : 'mean', 'value_eur' : 'sum'}).rename(columns={'overall' : 'overall_17', 'value_eur' : 'value_eur_17'}).reset_index()
club_level_18 = db_18.groupby(['league_name','club_name']).agg({'overall' : 'mean', 'value_eur' : 'sum'}).rename(columns={'overall' : 'overall_18', 'value_eur' : 'value_eur_18'}).reset_index()
club_level_19 = db_19.groupby(['league_name','club_name']).agg({'overall' : 'mean', 'value_eur' : 'sum'}).rename(columns={'overall' : 'overall_19', 'value_eur' : 'value_eur_19'}).reset_index()
club_level_20 = db_20.groupby(['league_name','club_name']).agg({'overall' : 'mean', 'value_eur' : 'sum'}).rename(columns={'overall' : 'overall_20', 'value_eur' : 'value_eur_20'}).reset_index()
club_level_21 = db_21.groupby(['league_name','club_name']).agg({'overall' : 'mean', 'value_eur' : 'sum'}).rename(columns={'overall' : 'overall_21', 'value_eur' : 'value_eur_21'}).reset_index()

clubs_all_years = [club_level_15, club_level_16, club_level_17, club_level_18, club_level_19, club_level_20, club_level_21]

clubs = reduce(lambda  left,right: pd.merge(left,right,on=['league_name','club_name'],
                                            how='outer'), clubs_all_years)

#clubs level 
column_value = ['value_eur_15','value_eur_16','value_eur_17','value_eur_18','value_eur_19','value_eur_20', 'value_eur_21']

for i in column_value:
    clubs[i] = clubs.apply(lambda x: "{:,}".format(x[i]), axis=1)

clubs_level = clubs.drop(columns=column_value)
clubs_level = clubs_level.rename(columns = {'overall_15' : 2015, 'overall_16' : 2016, 'overall_17' : 2017, 'overall_18' : 2018, 'overall_19' : 2019, 'overall_20' : 2020, 'overall_21' : 2021})
clubs_level = clubs_level.melt(['league_name','club_name'], var_name='year').rename(columns = {'value':'team_level'})

#clubs value
column_overall = ['overall_15', 'overall_16', 'overall_17', 'overall_18', 'overall_19', 'overall_20', 'overall_21']

clubs_value = clubs.drop(columns=column_overall)
clubs_value = clubs_value.rename(columns = {'value_eur_15' : 2015, 'value_eur_16' : 2016, 'value_eur_17' : 2017, 'value_eur_18' : 2018, 'value_eur_19' : 2019, 'value_eur_20' : 2020, 'value_eur_21' : 2021})
clubs_value = clubs_value.melt(['league_name','club_name'], var_name='year').rename(columns = {'value':'team_value'})

#clubs wages
#TO BE ADDED

#gathering clubs level and value datasets
clubs_df = [clubs_level, clubs_value]
clubs_vf = reduce(lambda  left,right: pd.merge(left,right,on=['league_name', 'club_name', 'year'], how='outer'), clubs_df)


# leagues / clubs / players datasets with all characteristics (2021)
game_characteristics = [
       'potential',
       'skill_moves', 'pace', 'shooting',
       'passing', 'dribbling', 'defending', 'physic', 'gk_diving',
       'gk_handling', 'gk_kicking', 'gk_reflexes', 'gk_speed',
       'gk_positioning', 'attacking_crossing', 'attacking_finishing',
       'attacking_heading_accuracy', 'attacking_short_passing',
       'attacking_volleys', 'skill_dribbling', 'skill_curve',
       'skill_fk_accuracy', 'skill_long_passing', 'skill_ball_control',
       'movement_acceleration', 'movement_sprint_speed', 'movement_agility',
       'movement_reactions', 'movement_balance', 'power_shot_power',
       'power_jumping', 'power_stamina', 'power_strength', 'power_long_shots',
       'mentality_aggression', 'mentality_interceptions',
       'mentality_positioning', 'mentality_vision', 'mentality_penalties',
       'mentality_composure', 'defending_marking', 'defending_standing_tackle',
       'defending_sliding_tackle', 'goalkeeping_diving',
       'goalkeeping_handling', 'goalkeeping_kicking',
       'goalkeeping_positioning', 'goalkeeping_reflexes']

db_21_characteristics_clubs = db_21[game_characteristics + ['club_name', 'league_name']].groupby(['league_name','club_name']).mean().reset_index()