import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(
            page_title="Bulk Football Insights",
            page_icon="⚽💡",
            layout="centered",
            initial_sidebar_state="auto")

st.sidebar.title('Bulk Football Insights ⚽💡')

df_dimension = pd.DataFrame({'first column': ['Select a dimension of analysis', 'Leagues', 'Intra-league', 'Clubs', 'Players']})
dimension_analysis = st.sidebar.selectbox('Select a dimension of analysis', df_dimension['first column'])

from FIFA_datasets import best_evol_champ
from FIFA_datasets import map_df
from FIFA_datasets import top_5
from FIFA_datasets import clubs_value

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

if dimension_analysis == 'Intra-league':
    fig_evol_level_clubs = px.line(clubs_value[clubs_value['league_name'] == 'French Ligue 1'],
                                 x="year", 
                                 y="team_value", 
                                 color='club_name',
                                 title='French clubs average value evolution')
    st.plotly_chart(fig_evol_level_clubs)