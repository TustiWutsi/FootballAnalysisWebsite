import streamlit as st
import pandas as pd
import plotly.express as px

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

if analysis_choice == 'FIFA datasets analyses':

    df_dimension = pd.DataFrame({'first column': ['Select a dimension of analysis', 'Leagues', 'Clubs', 'Players']})
    dimension_analysis = st.selectbox('Go to', df_dimension['first column'])

    from FIFA_datasets import best_evol_champ
    from FIFA_datasets import map_df
    from FIFA_datasets import top_5
    from FIFA_datasets import clubs_value
    from FIFA_datasets import clubs_level
    from FIFA_datasets import clubs_vf

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
        st.markdown('''# **Intra-league focus**''')

        level_or_value = st.radio('Select the focus you are interested in', ('Average players level', 'Average players value'))

        league_choice = st.selectbox('Select a league', list(clubs_vf.league_name.unique()))

        if level_or_value == 'Average players value':
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

        col1, col2, col3 = st.columns(3)
        col1.metric("SPDR S&P 500", "$437.8", "-$1.25")
        col2.metric("FTEC", "$121.10", "0.46%")
        col3.metric("BTC", "$46,583.91", "+4.87%")

        st.markdown('''# **Specific clubs focus**''')

        teams_to_compare = st.multiselect('Pick your teams', clubs_vf['club_name'].unique())

        if level_or_value == 'Average players value':
            fig_evol_value_clubs = px.line(clubs_value[clubs_value['club_name'].isin(teams_to_compare)],
                                        x="year", 
                                        y="team_value", 
                                        color='club_name',
                                        title= f'Clubs average players value evolution')
            st.plotly_chart(fig_evol_value_clubs)
        if level_or_value == 'Average players level':
            fig_evol_level_clubs = px.line(clubs_level[clubs_level['club_name'].isin(teams_to_compare)],
                                        x="year", 
                                        y="team_level", 
                                        color='club_name',
                                        title= f'Clubs average players level evolution')
            st.plotly_chart(fig_evol_level_clubs)

        #display top 3 stats of selected teams