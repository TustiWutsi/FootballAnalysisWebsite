import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap
import numpy as np
import time

from front import header
from front import in_progress

from FIFA_datasets import top_5
from FIFA_datasets import clubs_value
from FIFA_datasets import best_evol_champ
from FIFA_datasets import map_df
from FIFA_datasets import clubs
from FIFA_datasets import clubs_level
from FIFA_datasets import clubs_vf
from FIFA_datasets import increase_clubs_plot
from FIFA_datasets import decrease_clubs_plot
from FIFA_datasets import game_characteristics
from FIFA_datasets import db_21_characteristics_clubs
from FIFA_datasets import radar_plot
from FIFA_datasets import bar_plot

from clustering import strikers_num_scaled_and_transformed
from clustering import labelling
from clustering import strikers_with_label
from clustering import strikers_clusters
from clustering import clubs_with_clusters
from clustering import kms_clubs
from clustering import db_22_vf
from clustering import game_characteristics_22
from clustering import game_characteristics_22_gk
from clustering import minmax
from clustering import pca
from clustering import elbow_chart
from clustering import cluster_viz
from clustering import interpret_clusters

from pitch_analysis import world_cup_games
from pitch_analysis import goal_number
from pitch_analysis import actions

from mplsoccer.statsbomb import read_event, EVENT_SLUG
from mplsoccer import Pitch, VerticalPitch
from mplsoccer.cm import create_transparent_cmap
from mplsoccer.scatterutils import arrowhead_marker
from mplsoccer.statsbomb import read_event, EVENT_SLUG
from mplsoccer.utils import FontManager

st.set_page_config(
            page_title="Bulk Football Insights",
            page_icon="⚽💡",
            layout="centered",
            initial_sidebar_state="auto")

st.sidebar.title('Bulk Football Insights ⚽💡')

st.sidebar.markdown("""
                    
                    Here you will find :
                    - Some insights about leagues, clubs and players from FIFA datasets
                    - Players and teams clustering
                    - Teams performance predictions
                    - Event-data analyses
                    - ... \n
                    **In short, a brief overview of what we can do with open-source football data 📈🔎 !**
                    
                    # **Navigation**

                    """)

df_analysis = pd.DataFrame({'first column': ['Choose a type of analysis', 'FIFA datasets analyses', 'Teams & players clustering', 'Teams performance prediction', 'Event-data vizualisations', 'Fans sentiment analysis']})
analysis_choice = st.sidebar.selectbox('Go to', df_analysis['first column'])

### FIFA DATASETS ANALYSES ###
if analysis_choice == 'FIFA datasets analyses':

    header("This section contains basic data analyses with FIFA players datasets, from 2015 to 2021, at 3 levels : leagues, clubs and players. You will get insights about level/value evolution, most significative improvements/decreases, or also game attributes comparison")
    st.text("")

    df_dimension = pd.DataFrame({'first column': ['Select a dimension of analysis', 'Leagues', 'Clubs', 'Players']})
    dimension_analysis = st.selectbox('Go to', df_dimension['first column'])

    if dimension_analysis == 'Leagues':
        
        in_progress()
        st.text("")

        fig_map = px.scatter_geo(map_df, 
                                lat="latitude",
                                lon="longitude", 
                                size="overall_21", title="World map with leagues global level",
                                hover_data=['def_21','mid_21','str_21','gk_21'])
        st.plotly_chart(fig_map)   

        fig_best_evol_champ = px.bar(best_evol_champ,
                                    x='league_name',
                                    y='overall_evol', 
                                    color='overall_20', 
                                    title='Leagues where global level has most increased')
        st.plotly_chart(fig_best_evol_champ)

        fig_top_5_evol = px.line(top_5, 
                                x="year",
                                y="league_level",
                                color='league_name',
                                title='Top 5 european leagues global level evolution')
        st.plotly_chart(fig_top_5_evol)
        # add 'hover_data' with position level

    if dimension_analysis == 'Clubs':
        
        st.header('''Intra-league focus''')
        
        league_choice = st.selectbox('Select a league', list(clubs_vf.league_name.unique()))
        
        st.subheader('''Evolution of level / value / wages''')

        level_or_value = st.radio('Select the focus you are interested in', ('Average players level', 'Total players value'))
        
        if level_or_value == 'Total players value':
            fig_evol_value_clubs = px.line(clubs_value[clubs_value['league_name'] == league_choice],
                                        x="year", 
                                        y="team_value", 
                                        color='club_name',
                                        title= f'Clubs average players value evolution in {league_choice}')
            st.plotly_chart(fig_evol_value_clubs)
        if level_or_value == 'Average players level':
            fig_evol_level_clubs = px.line(clubs_level[clubs_level['league_name'] == league_choice],
                                        x="year", 
                                        y="team_level", 
                                        color='club_name',
                                        title= f'Clubs average players level evolution in {league_choice}')
            st.plotly_chart(fig_evol_level_clubs)
        st.markdown('''*TIPS : You can double-clik on teams to keep only them on the chart* **🖱**''')

        st.subheader('''Game characteristics ranking''')

        characteristic_choice = st.selectbox('Select a game characteristic', game_characteristics)

        fig_top_characteristics_intra_clubs = px.bar(db_21_characteristics_clubs[db_21_characteristics_clubs['league_name'] == league_choice].sort_values(characteristic_choice, ascending=False), x='club_name', y=characteristic_choice)
        fig_top_characteristics_intra_clubs.update_layout(yaxis_range=[round(db_21_characteristics_clubs[characteristic_choice].min()),round(db_21_characteristics_clubs[characteristic_choice].max())])
        st.plotly_chart(fig_top_characteristics_intra_clubs)

        st.header('''Specific clubs focus''')

        teams_to_compare = st.multiselect('Pick your teams', clubs_vf['club_name'].unique())

        st.subheader('''Evolution of level / value / wages''')

        level_or_value_2 = st.radio('Choose the focus you are interested in', ('Average players level', 'Total players value'))

        if level_or_value_2 == 'Total players value':
            fig_evol_value_clubs = px.line(clubs_value[clubs_value['club_name'].isin(teams_to_compare)],
                                        x="year", 
                                        y="team_value", 
                                        color='club_name',
                                        title= f'Clubs average players value evolution')
            st.plotly_chart(fig_evol_value_clubs)
        if level_or_value_2 == 'Average players level':
            fig_evol_level_clubs = px.line(clubs_level[clubs_level['club_name'].isin(teams_to_compare)],
                                        x="year", 
                                        y="team_level", 
                                        color='club_name',
                                        title= f'Clubs average players level evolution')
            st.plotly_chart(fig_evol_level_clubs)

        st.subheader('''Game characteristics ranking''')

        characteristic_choice_2 = st.multiselect('Choose a game characteristic', game_characteristics)

        bar_plot(characteristic_choice_2, teams_to_compare)

        st.text("")
        st.markdown('3) You can also compare game attributes with a radar viz :')
        radar_plot(characteristic_choice_2, teams_to_compare)

        st.text("")
        st.markdown('4) Have a look to the clubs that have most improved/declined in last years :')
        level_or_value_3 = st.radio('Choose the focus you are interested in', ('Teams Level', 'Teams Value'))
        top_number = st.slider('Select the number of teams', 1, 30, 10)

        increase_clubs_plot(level_or_value_3, top_number)

        st.text("")
        decrease_clubs_plot(level_or_value_3, top_number)

    if dimension_analysis == 'Players':
        in_progress()
        st.text("")

### CLUSTERING ###
if analysis_choice == 'Teams & players clustering':

    header("In this part, you can highlight the main families/clusters of players for each position on the pitch, based on FIFA22 players' attributes. For a few positions, it will be hard for the model to identify other patterns than good vs bad players, but for most of them it is insightful !")
    st.text("")

    df_clustering = pd.DataFrame({'first column': ['Select an option', 'Teams clustering', 'Players clustering']})
    clustering_choice = st.selectbox('What king of clustering analysis are you interested in ?', df_clustering['first column'])

    if clustering_choice == 'Players clustering':
        st.text("")
        st.markdown('1) First, select a position you are interested in')
        position_selected = st.selectbox('Choose a position', db_22_vf['club_position'].unique())
        
        st.text("")
        st.markdown('2) Then, choose the most appropriate number of clusters according to the chart below, following the elbow method')
        df_position_selected = db_22_vf[db_22_vf['club_position'] == position_selected]
        if position_selected == 'GK':
            df_position_selected_2 = db_22_vf[db_22_vf['club_position'] == position_selected][game_characteristics_22_gk]
        else:
            df_position_selected_2 = db_22_vf[db_22_vf['club_position'] == position_selected][game_characteristics_22]

        df_scaled = minmax(df_position_selected_2)
        df_scaled_transformed = pca(df_scaled)
        elbow_chart(df_scaled_transformed)
        cluster_number = st.number_input('Insert the chosen number of clusters')

        st.text("")
        st.markdown('3) Now, launch the training of the model (Kmeans) and visualize/interpret your clusters')
        if st.checkbox('Launch training'):
            cluster_viz(cluster_number, df_scaled_transformed)
        if st.checkbox('Interpret clusters'):
            feature_number = st.slider('Select the number of most significative features for each cluster ', 1, 10, 5)
            players_with_labels = interpret_clusters(cluster_number, feature_number, df_scaled, df_position_selected)

            st.text("")
            st.markdown('4) Select a player from this position to know to which cluster he belongs')
            player_selected_clustering = st.selectbox('Choose a player', players_with_labels['short_name'].unique())
            player_selected_cluster_number = pd.DataFrame(players_with_labels[players_with_labels['short_name'] == player_selected_clustering].label).iloc[0,0]
            
            col1, col2 = st.columns(2)
            col1.image(pd.DataFrame(players_with_labels[players_with_labels['short_name'] == player_selected_clustering].player_face_url).iloc[0,0], width=200)
            col2.text("")
            col2.markdown(f'{player_selected_clustering} belongs to cluster n°**{player_selected_cluster_number}**')
            st.text("")
            st.markdown(f'Find below some players similar to {player_selected_clustering}, and the top 10 features that most defined this cluster ')
            st.write(players_with_labels[players_with_labels['label'] == player_selected_cluster_number][['short_name', 'club_name', 'overall', 'value_eur']].sort_values('overall', ascending=False))


    if clustering_choice == 'Teams clustering':
        
        df_club_selection = pd.DataFrame({'first_column' : ['Select a club'] + list(clubs_with_clusters['club_name'].unique())})
        club_selection = st.selectbox('Which club are you interested in ?', df_club_selection['first_column'])

        selected_club_cluster = pd.DataFrame(clubs_with_clusters[clubs_with_clusters['club_name'] == club_selection].club_cluster).iloc[0,0]

        st.markdown(f'**{club_selection}** belongs to cluster n°**{selected_club_cluster}** :')
        st.text("")
        st.markdown(f'Find below some clubs similar to **{club_selection}** : ')

        st.write(clubs_with_clusters[clubs_with_clusters['club_cluster'] == selected_club_cluster][['club_name']])
        st.text("")
        st.markdown(f'Find below the top 10 game characteristics associated to cluster n°**{selected_club_cluster}** : ')

        fig, ax = plt.subplots(figsize=(14,6))
        sns.barplot(x="Feature", y="Weight", data=pd.DataFrame(kms_clubs.feature_importances_[selected_club_cluster][:10], columns=["Feature", "Weight"]))
        plt.xticks(rotation=-45, ha="left");
        ax.tick_params(axis='both', which='major', labelsize=22)
        plt.title(f'Highest Weight Features in Cluster {selected_club_cluster}', fontsize='xx-large')
        plt.xlabel('Feature', fontsize=18)
        plt.ylabel('Weight', fontsize=18)

        st.pyplot(fig)

### PITCH DATA ANALYSES ###
if analysis_choice == 'Event-data vizualisations':

    header("You will find here a very quick overview of what can be done with football 'event-data' ( = action information and ball location(x,y) at each second of a game) for 2018 World Cup games. This open-data is provided by Statsbomb")
    st.text("")
    
    #Game selection
    df_game_pitch_selection = pd.DataFrame({'first_column' : ['Select a game'] + list(world_cup_games['game'].unique())})
    game_pitch_selection = st.selectbox('In which World Cup 2018 game are you interested in ?', df_game_pitch_selection['first_column'])

    selected_game_id = pd.DataFrame(world_cup_games[world_cup_games['game'] == game_pitch_selection].match_id).iloc[0,0]

    kwargs = {'related_event_df': False, 'shot_freeze_frame_df': False,'tactics_lineup_df': False, 'warn': False}
    df = read_event(f'{EVENT_SLUG}/{selected_game_id}.json', **kwargs)['event'] 

    home_team = pd.DataFrame(world_cup_games[world_cup_games['match_id'] == selected_game_id].home_team).iloc[0,0]
    away_team = pd.DataFrame(world_cup_games[world_cup_games['match_id'] == selected_game_id].away_team).iloc[0,0]

    home_score = pd.DataFrame(world_cup_games[world_cup_games['match_id'] == selected_game_id].home_score).iloc[0,0]
    away_score = pd.DataFrame(world_cup_games[world_cup_games['match_id'] == selected_game_id].away_score).iloc[0,0]

    #Score of the game
    #col1, col2 = st.columns(2)
    #col1.metric(f'{home_team}', home_score)
    #col2.metric(f'{away_team}', away_score)
    #st.text("")

    #Shots mapping on pitch
    st.text("")
    st.markdown('1) Shots mapped on the pitch, with associated xG :')

    team_pitch = st.radio('Select a team', df.team_name.unique())

    df_shots = df[(df.type_name == 'Shot') & (df.team_name == team_pitch)].copy()
    
    df_pass = df[(df.type_name == 'Pass') &
                   (df.team_name == team_pitch) &
                   (~df.sub_type_name.isin(['Throw-in', 'Corner', 'Free Kick', 'Kick Off']))].copy()

    fm_rubik = FontManager(('https://github.com/google/fonts/blob/main/ofl/rubikmonoone/'
                        'RubikMonoOne-Regular.ttf?raw=true'))
    

    # filter goals / non-shot goals
    df_goals = df_shots[df_shots.outcome_name == 'Goal'].copy()
    df_non_goal_shots = df_shots[df_shots.outcome_name != 'Goal'].copy()

    # Preparing player name and expecting goals for labels
    players_name = list(df_shots['player_name'])
    xg = list(round(df_shots['shot_statsbomb_xg'], 2))

    pitch_shot = Pitch()

    fig, ax = pitch_shot.draw(figsize=(20, 15))

    # plot non-goal shots with hatch
    sc1 = pitch_shot.scatter(df_non_goal_shots.x, df_non_goal_shots.y,
                        # size varies between 100 and 1900 (points squared)
                        s=(df_non_goal_shots.shot_statsbomb_xg * 1900) + 100,
                        edgecolors='#b94b75',  # give the markers a charcoal border
                        c='None',  # no facecolor for the markers
                        hatch='///',  # the all important hatch (triple diagonal lines)
                        # for other markers types see: https://matplotlib.org/api/markers_api.html
                        marker='o',
                        ax=ax)

    # plot goal shots with a football marker
    # 'edgecolors' sets the color of the pentagons and edges, 'c' sets the color of the hexagons
    sc2 = pitch_shot.scatter(df_goals.x, df_goals.y,
                        # size varies between 100 and 1900 (points squared)
                        s=(df_goals.shot_statsbomb_xg * 1900) + 100,
                        edgecolors='#b94b75',
                        linewidth=0.6,
                        c='white',
                        marker='football',
                        ax=ax)

    txt = ax.text(x=50, y=40, s=f'{home_team} shots\nversus {away_team}',
                size=40,
                # here i am using a downloaded font from google fonts instead of passing a fontdict
                fontproperties=fm_rubik.prop, color=pitch_shot.line_color,
                va='center', ha='center')

    for i, txt in enumerate(list(zip(players_name, xg))):
        ax.annotate(f'    {txt}', (list(df_shots.x)[i], list(df_shots.y)[i]), fontsize=15)
        
    st.pyplot(fig)
    
    #Total xG and xG evolutions during game
    st.text("")
    st.markdown('2) Total xG and xG evolutions during game :')

    col1, col2 = st.columns(2)
    col1.metric(f'{home_team} total xG', round(df[df['team_name'] == home_team]['shot_statsbomb_xg'].sum(),2))
    col2.metric(f'{away_team} total xG', round(df[df['team_name'] == away_team]['shot_statsbomb_xg'].sum(),2))

    df_xg = df.groupby(['team_name', 'minute'])[['shot_statsbomb_xg']].sum().groupby(level=0).cumsum().reset_index()

    fig_xg = px.line(df_xg, x="minute", y="shot_statsbomb_xg", color='team_name')
    st.plotly_chart(fig_xg)

    #Goals actions
    st.text("")
    st.markdown('3) Goals actions mapped on the pitch:')

    df_goal_nb = goal_number(df)
    
    df_goal_selection = pd.DataFrame({'first_column' : ['Select a goal'] + list(df_goal_nb['goal_number'].unique())})
    goal_selected = st.selectbox('Select a goal ?', df_goal_selection['first_column'])

    index_goal = df_goal_nb[df_goal_nb['goal_number'] == goal_selected].index[0]
    df_action_goal = df_goal_nb.iloc[index_goal-4:index_goal+1,:]

    actions(
    location=df_action_goal[["x", "y", "end_x", "end_y"]],
    action_type=df_action_goal.type_name,
    team=df_action_goal.team_name,
    result= df_action_goal.outcome_name == "goal",
    label=df_action_goal[["minute", "type_name", "player_name", "team_name"]],
    labeltitle=["time","actiontype","player","team"],
    zoom=False
    )

### TEAMS PERFORMANCE PREDICTION
if analysis_choice == 'Teams performance prediction':
    in_progress()

### FANS SENTIMENT ANALYSIS
if analysis_choice == 'Fans sentiment analysis':
    in_progress()