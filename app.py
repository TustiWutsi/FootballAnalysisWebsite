import streamlit as st
import pandas as pd
import plotly.express as px

from FIFA_datasets import top_5
from FIFA_datasets import clubs_value
from FIFA_datasets import best_evol_champ
from FIFA_datasets import map_df
from FIFA_datasets import clubs_level
from FIFA_datasets import clubs_vf
from FIFA_datasets import game_characteristics
from FIFA_datasets import db_21_characteristics_clubs

from clustering import strikers_num_scaled_and_transformed
from clustering import labelling
from clustering import strikers_with_label
from clustering import strikers_clusters

st.set_page_config(
            page_title="Bulk Football Insights",
            page_icon="⚽💡",
            layout="centered",
            initial_sidebar_state="auto")

st.sidebar.title('Bulk Football Insights ⚽💡')

st.sidebar.markdown("""
                    
                    Here you will find :
                    - Some insights about leagues, clubs and players from FIFA datasets
                    - Teams and players clustering
                    - Past games statistics and future games predictions
                    - Pitch data analysis
                    - ... \n
                    **In short, a brief overview of what we can do with open-source football data 📈🔎 !**
                    
                    # **Navigation**

                    """)

df_analysis = pd.DataFrame({'first column': ['Choose a type of analysis', 'FIFA datasets analyses', 'Teams & Players Clustering', 'Games Stats & Prediction', 'Pitch Data analyses']})
analysis_choice = st.sidebar.selectbox('Go to', df_analysis['first column'])

### FIFA DATASETS ANALYSES ###
if analysis_choice == 'FIFA datasets analyses':

    df_dimension = pd.DataFrame({'first column': ['Select a dimension of analysis', 'Leagues', 'Clubs', 'Players']})
    dimension_analysis = st.selectbox('Go to', df_dimension['first column'])

    if dimension_analysis == 'Leagues':

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

        #col1, col2, col3 = st.columns(3)
        #col1.metric("SPDR S&P 500", "$437.8", "-$1.25")
        #col2.metric("FTEC", "$121.10", "0.46%")
        #col3.metric("BTC", "$46,583.91", "+4.87%")

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

        characteristic_choice_2 = st.selectbox('Choose a game characteristic', game_characteristics)

        fig_top_characteristics_cross_clubs = px.bar(db_21_characteristics_clubs[db_21_characteristics_clubs['club_name'].isin(teams_to_compare)].sort_values(characteristic_choice_2, ascending=False),
                                                    x='club_name', 
                                                    y=characteristic_choice_2,
                                                    title= f'Ranking of selected clubs for {characteristic_choice_2}')
        fig_top_characteristics_cross_clubs.update_layout(yaxis_range=[round(db_21_characteristics_clubs[characteristic_choice_2].min()),round(db_21_characteristics_clubs[characteristic_choice_2].max())])
        st.plotly_chart(fig_top_characteristics_cross_clubs)


### CLUSTERING ###
if analysis_choice == 'Teams & Players Clustering':

    clustering_choice = st.selectbox('Select the type of clustering you are interested in', ['Teams clustering', 'Players clustering'])

    if clustering_choice == 'Players clustering':
        
        position_choice = st.radio('Select a position', ('Defenders', 'Midfielders', 'Strikers'))

        if position_choice == 'Strikers':
            player_selection = st.selectbox('Choose a player', strikers_clusters['short_name'].unique())

            selected_player_cluster = pd.DataFrame(strikers_clusters[strikers_clusters['short_name'] == player_selection].label).iloc[0,0]
            selected_player_description = pd.DataFrame(strikers_clusters[strikers_clusters['short_name'] == player_selection].cluster_description).iloc[0,0]

            st.markdown(f'{player_selection} belongs to cluster n°**{selected_player_cluster}** :')
            st.markdown(f'*{selected_player_description}*')
            st.markdown(f'Find below some players similar to {player_selection} : ')

            #selected_value = st.slider('You can select a maximum player value', 0, 100000000, 100000000 )
            selected_value = st.number_input('You can type a maximum player value to filter similar players')


            st.write(strikers_clusters[(strikers_clusters['label'] == selected_player_cluster) & (strikers_clusters['value_eur'] < selected_value)][['short_name','value_eur']])

            if st.button('click to vizualize strikers clusters'):
                fig_strikers_clusters = px.scatter_3d(strikers_num_scaled_and_transformed,x=0,y=1,z=2,color=labelling)
                st.plotly_chart(fig_strikers_clusters)
