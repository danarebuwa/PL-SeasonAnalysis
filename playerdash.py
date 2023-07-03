#creating a player dashboard for arsenal players this season
import streamlit as st 
import pandas as pd
import numpy as np
import matplotlib 
import matplotlib.pyplot as plt
matplotlib.use("Agg")
from matplotlib.backends.backend_agg import RendererAgg
from matplotlib.figure import Figure
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import plotly.figure_factory as ff
import plotly.tools as tls
import requests 
from scipy import stats
import math
from mplsoccer import PyPizza, add_image, FontManager
from bs4 import BeautifulSoup
import json
from os.path  import basename
from soccerplots.radar_chart import Radar
from urllib.request import urlopen
from PIL import Image


_lock = RendererAgg.lock
plt.style.use('default')


st. set_page_config(layout="wide")
#df = pd.read_csv('arsenal_stats_cleaned.csv')

pl_df = pd.read_csv('pl_stats_cleaned.csv')
#create a dataframe that only shows the following columns "G-PK", "npxG","xA"
pi_attack = pl_df[['Player','Pos','G-PK', 'npxG','xA']]
#in pi_attack player column remove anything after "\"
pi_attack['Player'] = pi_attack['Player'].str.split("\\").str[0]
#pi_attack

pi_pass = pd.read_csv('pl_pass_stats.csv')
pi_pass = pi_pass[['Player','Pos','Prog', 'TCmp%']]
#in pi_pass player column remove anything after "\"
pi_pass['Player'] = pi_pass['Player'].str.split('\\').str[0]
#pi_pass 

pi_poss = pd.read_csv('pl_possesion.csv')
#in pi_poss player column remove anything after "\"
pi_poss['Player'] = pi_poss['Player'].str.split('\\').str[0]
pi_poss = pi_poss[['Player','Pos','CProg', '1/3']]
#pi_poss

pi_def = pd.read_csv('pl_defense.csv')
pi_def = pi_def[['Player','Pos','%', 'Int','Press','TklW']]
#in pi_def player column remove anything after "\"
pi_def['Player'] = pi_def['Player'].str.split('\\').str[0]
#pi_def

#merge the following dataframes pi_attack, pi_pass, pi_poss, pi_def
pi_df = pd.merge(pi_attack, pi_pass, on='Player')
#drop the duplicate column
pi_df = pi_df.drop(columns=['Pos_x'])
pi_df = pi_df.rename(columns={'Pos_y':'Pos'})
pi_df = pd.merge(pi_df, pi_poss, on='Player')
#drop the duplicate column
pi_df = pi_df.drop(columns=['Pos_x'])
pi_df = pi_df.rename(columns={'Pos_y':'Pos'})
pi_df = pd.merge(pi_df, pi_def, on='Player')
#drop the duplicate column
pi_df = pi_df.drop(columns=['Pos_x'])
pi_df = pi_df.rename(columns={'Pos_y':'Pos'})
#delete duplicate rows
pi_df = pi_df.drop_duplicates()
#make Pos the second column
pi_df = pi_df[['Player','Pos','G-PK', 'npxG','xA','Prog', 'TCmp%','CProg', '1/3','%', 'Int','Press','TklW']]


#remove anything with "\" and anything after it in defensivedf
#defensivedf = defensivedf[defensivedf['Player'].str.contains("\") == False]

#in pl_df player column remove anything after "\"
pl_df['Player'] = pl_df['Player'].str.split("\\").str[0]


#in pl_df nation column remove anything before the first capital letter
pl_df['Nation'] = pl_df['Nation'].str.split(" ").str[0]

#make all pl_df nation capitalized
pl_df['Nation'] = pl_df['Nation'].str.upper()
#pl_df

#make a data frame with squad just "arsenal" from pl_df
df = pl_df[pl_df['Squad'].str.contains('Arsenal')]

# Calculate the averages based on the player's position for columns starting from the 8th one
average_columns = pi_df.columns[7:]
pi_df_avg = pi_df[['Pos'] + list(average_columns)]
#player_df_avg = pi_df_avg.groupby('Pos').mean().reset_index()
columns_to_drop = ['Rk','Age','90s','Born']
player_df_avg = player_df_avg.drop(columns=[col for col in columns_to_drop if col in player_df_avg.columns], axis=1)


#drop the following columns from player_df_avg "Rk", "Age", "90s","Pos","Pk","PKatt"
player_df_avg = player_df_avg.drop(columns=['Rk','Age','90s','Born'], axis=1)
#create a column in player_avg_df called "Player"
player_df_avg['Player'] = 'Average'

st.markdown("<style> .reportview-container .main .block-container { width: 100%; } </style>", unsafe_allow_html=True)
#create a sidebar that allows you to select a player and season to view
st.sidebar.header("Player Dashboard")

#in sidebar "select a team" dropdown menu from pl_df
team = st.sidebar.selectbox('Select a Team',(pl_df['Squad'].unique()))
#in sidebar "select a player", create a dropdown list of all players in selected team
player = st.sidebar.selectbox('Select a Player',(pl_df[pl_df['Squad'] == team]['Player'].unique()))

#player = st.sidebar.selectbox('Select a Player',(df['Player'].unique()))

row1_spacer1, row1_1, row1_spacer2, row1_2, row1_spacer3 = st.columns(
    (.1, 2, 1.5, 1, .1)
    )
#add a header to the main page
with row1_1:
    st.markdown('<h1 style="font-size: 1.9rem;">Player Season Review 2021/22 </h1>', unsafe_allow_html=True)


#on the same row as the header, add a subheader "by Daniel Bosun-Arebuwa"
#with row1_2:
    #st.write('')
    #row1_2.subheader(
    #'A Web App by [Daniel Bosun-Arebuwa](https://twitter.com/chambs_dc/)')

#create a second row
row2_1, row2_spacer1, row2_2 = st.columns(
    (.9, .3, 1.6)
    )

#add player image from the same folder as playerdash.py
#with row2_1:
    #st.image(f'{player}.png', width=200)

    
#in the second row for the selected player, create a colum showing the player, nation, pos, age
with row2_1, _lock:
    player2 = pl_df[pl_df['Player'] == player]
    st.write(f'Name: {player2["Player"].iloc[0]}')
    st.write(f'Nation: {player2["Nation"].iloc[0]}')
    st.write(f'Position: {player2["Pos"].iloc[0]}')
    st.write(f'Age: {player2["Age"].iloc[0]}')

#in the second row show player heatmap
# ...

with row2_2:
    st.write('Player Performance Radar')

    player_df = pi_df[pi_df['Pos'] == player2['Pos'].iloc[0]]

    player3 = player_df[player_df['Player'] == player]

    params = list(player_df.columns)
    params = params[2:]

    players = player_df.loc[player_df['Player']== player].reset_index()
    players = list(players.loc[0])
    players = players[3:]

    values = []
    for x in range(len(params)):
        if players[x] is not None:
            percentile_score = stats.percentileofscore(player_df[params[x]],players[x])
            if math.isnan(percentile_score):
                values.append(0)
            else:
                values.append(math.floor(percentile_score))
        else:
            values.append(0)  # Replace None with 0. Adjust as needed for your context.

    for n,i in enumerate(values):
        if i == 100:
            values[n] = 99

    baker = PyPizza(
        params=params,
        straight_line_color="#000000",
        straight_line_lw=1,
        last_circle_lw=1,
        other_circle_lw=1,
        other_circle_ls="-."
    )

    fig, ax = baker.make_pizza(
        values,
        figsize=(7, 5),
        param_location=110,
        kwargs_slices=dict(
            facecolor="#6CABDD",
            edgecolor="#000000",
            zorder=2,
            linewidth=1
        ),
    # ... Rest of your code

    kwargs_params=dict(
        color="#000000",
        fontsize=12,
        va="center",
        alpha=.5
    ),
    kwargs_values=dict(
        color="#000000",
        fontsize=12,
        zorder=3,
        bbox=dict(
            edgecolor="#000000",
            facecolor="#6CABDD",
            boxstyle="round,pad=0.2",
            lw=1
        )
    )
)

    # add title
    fig.text(
        0.515, 0.97, player , size=18,
        ha="center", color="#000000"
    )

    # add subtitle
    fig.text(
        0.515, 0.942,
        "Per 90 Percentile Rank | 2021-22",
        size=15,
        ha="center", color="#000000"
    )

    # add credits
    notes = 'Players only with more than 15 90s'
    CREDIT_1 = "data: statsbomb via fbref"
    CREDIT_2 = "inspired by: @Worville, @FootballSlices, @somazerofc & @Soumyaj15209314"

    fig.text(
        0.99, 0.005, f"{notes}\n{CREDIT_1}\n{CREDIT_2}", size=9,
        color="#000000",
        ha="right"
    )

    plt.savefig('pizza.png',dpi=500,bbox_inches = 'tight')
    st.pyplot(fig)




    #avg_df = pl_df.groupby('Pos').mean()
    #avg_df
    #show only selected player pos avg_df
    player_df_avg = pi_df_avg.groupby('Pos').mean()
    #make index of player_df_avg into a column
    player_df_avg.reset_index(inplace=True)
    #rename index to Pos
    player_df_avg.rename(columns={'index': 'Pos'}, inplace=True)

    #drop the followinfg columns from player_df "Rk", "Nation", "Squad", "Age", "90s","Pos","Pk","PKatt"
    #player_df = player_df.drop(columns=['Rk','Nation','Squad','Age','90s','Matches','Born'], axis=1)


    #drop the following columns from player_df_avg "Rk", "Nation", "Squad", "Age", "90s","Pos","Pk","PKatt"
    player_df_avg = player_df_avg.drop(columns=['Rk','Age','90s','Born'], axis=1)
    #create a column in player_avg_df called "Player"
    player_df_avg['Player'] = 'Average'
   
    #show the selected player Pos in player_df_avg
    player_df_avg = player_df_avg[player_df_avg['Pos'] == player_df['Pos'].iloc[0]]
    #player_df_avg



#create a third row
st.markdown('<h1 style="font-size: 1.5rem;">Scoring Potential </h1>', unsafe_allow_html=True)
row4_1, row4_spacer1, row4_2, row4_spacer2, row4_3 = st.columns(
    (1.6, .1, 1.6, .1, 1.6)
    )

#in the third row, add a header "Player Stats"
with row4_1, _lock:
    #reduce size of the subheader
    st.text('Player xG')
    #create a dataframe that shows players in the same position as the selected player
    df2_fw = pl_df[pl_df['Pos'] == player2['Pos'].iloc[0]]
    #in df2 create a row called "Avg" that is the average of all the columns
    df2_fw['AvgxG'] = df2_fw["xG"].mean()
    #single out the player you selected
    df2_fw = df2_fw[df2_fw['Player'] == player]
    #df2_fw
    #using a seaborn create a bar chart of player comparing "xG" to "AvgxG"
    # the x axis shows both the xG and AvgxG
    # the y axis shows the player
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3, 3), sharex=True)
    sns.barplot(x="xG", y="Player", data=df2_fw, ax=ax1)
    sns.barplot(x="AvgxG", data=df2_fw, ax=ax2)
    sns.despine(left=True)
    plt.show()
    st.pyplot(fig)

    #create a dataframe that shows players in the same position as the selected player
with row4_2, _lock:
    st.text('xG+xA vs G+A')
    df2_df = pl_df[pl_df['Pos'] == player2['Pos'].iloc[0]]
    #df2_df 

    #using seaborn create a scatter plot of "G+A" vs "xG+xA"
    #the x axis show "G+A"
    #the y axis show "xG+xA"
    #the color shows the player
    fig, (ax1) = plt.subplots(1, 1, figsize=(3, 3), sharex=True)
    sns.scatterplot(x="G+A", y="xG+xA", data=df2_df, ax=ax1)
    #highlight the player you selected
    ax1.scatter(df2_df["G+A"].iloc[0], df2_df["xG+xA"].iloc[0], color='red', s=100)

    sns.despine(left=True)
    plt.show()
    st.pyplot(fig)

    #create a swarm plot of "Gls.1" vs "xGls.1"
    #the x axis show "Gls.1"
    #the y axis show "xGls.1"
    #the color shows the player
with row4_3, _lock:
    st.text("Goals Per 90")
    fig, (ax1) = plt.subplots(1, 1, figsize=(3, 3), sharex=True)
    ax = sns.swarmplot(x="Gls.1", y="Pos", data=df2_df, ax=ax1)
    #highlight the player you selected
    ax1.scatter(df2_df["Gls.1"].iloc[0], df2_df["Pos"].iloc[0], color='red', s=100)
    #increase scale of x axis
    ax1.set_xlim(0, 1)
    sns.despine(left=True)
    ax.set_xlabel('Goals Per 90')
    plt.show()
    st.pyplot(fig)

#create a dataframe for player shooting stats  
df_shoot = pd.read_csv('pl_shooting_stats.csv')

#in df_shoot player column remove anything after "\"
df_shoot['Player'] = df_shoot['Player'].str.split("\\").str[0]

#in df_shoot nation column remove anything before the first capital letter
df_shoot['Nation'] = df_shoot['Nation'].str.split(" ").str[0]

#make all df_shoot nation capitalized
df_shoot['Nation'] = df_shoot['Nation'].str.upper()

#merge df_shoot and pl_df on player, nation, pos, age, xG, npxG
pl_df = pd.merge(pl_df, df_shoot, on=['Player', 'Nation', 'Pos', 'Age', 'xG', 'npxG', 'Rk', 'Squad', 'Born', '90s','Gls', 'PK','PKatt', 'Matches'])

st.markdown('<h1 style="font-size: 1.5rem;">Defense </h1>', unsafe_allow_html=True)
row3_1, row3_spacer1, row3_2, row3_spacer2, row3_3 = st.columns(
    (1,.1,1,.1,1)
)
pl_def = pd.read_csv('pl_defense.csv')
#in pl_def player column remove anything after "\"
pl_def['Player'] = pl_def['Player'].str.split("\\").str[0]
#pl_def

with row3_1, _lock:
    st.text('Defensive Tackle Completions')
    #create a dataframe that shows players in the same position as the selected player
    df2_def = pl_def[pl_def['Pos'] == player2['Pos'].iloc[0]]
    #in df2 create a row called "Avg" that is the average of the "Tkl%" columns
    df2_def['AvgTkl%'] = df2_def["Tkl%"].mean()
    #df2_def
    #single out the player you selected
    df2_def = df2_def[df2_def['Player'] == player]
    #using a seaborn create a bar chart of player comparing "Tkl%" to "AvgTkl"
    # the x axis shows both the Tkl% and AvgTkl
    # the y axis shows the player
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3, 3), sharex=True)
    sns.barplot(x="Tkl%", y="Player", data=df2_def, ax=ax1)
    sns.barplot(x="AvgTkl%", data=df2_def, ax=ax2)
    #change the title for x axis in ax2
    ax2.set_xlabel('Tkl%')
    ax2.set_ylabel('Average')
    #make the a axis title horizontal
    ax2.yaxis.set_label_coords(-0.1, 0.5)

    sns.despine(left=True)
    plt.show()
    st.pyplot(fig)
with row3_2, _lock:
    st.text('Defensive 3rd')
    df3_def = pl_def[pl_def['Pos'] == player2['Pos'].iloc[0]]
    #create a regplot of "Succ" and "Tkl+Int" in df2_def
    #the x axis show "Succ"
    #the y axis show "Tkl+Int"
    #the color shows the player
    fig, (ax1) = plt.subplots(1, 1, figsize=(3, 3), sharex=True)
    sns.regplot(x="Succ", y="Tkl+Int", data=df3_def, ax=ax1)
    #highlight the player you selected
    ax1.scatter(df3_def["Succ"].iloc[0], df3_def["Tkl+Int"].iloc[0], color='red', s=100)
    sns.despine(left=True)
    plt.show()
    st.pyplot(fig)

with row3_3, _lock:
    st.text('Pressures')
    #create a dataframe that shows players in the same position as the selected player
    df2_def = pl_def[pl_def['Pos'] == player2['Pos'].iloc[0]]

    #using seaborn create 3 swarm plots for "Def 3rd","Mid 3rd", and "Att 3rd" in df3_def
    #the x axis show "Def 3rd"
    #the y axis show "Pos"
    #the color shows the player 
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(3, 4), sharex=True)
    sns.swarmplot(x="Def 3rd", y="Pos", data=df3_def, ax=ax1)
    sns.swarmplot(x="Mid 3rd", y="Pos", data=df3_def, ax=ax2)
    sns.swarmplot(x="Att 3rd", y="Pos", data=df3_def, ax=ax3)
    #highlight the player you selected
    ax1.scatter(df3_def["Def 3rd"].iloc[0], df3_def["Pos"].iloc[0], color='red', s=100)
    ax2.scatter(df3_def["Mid 3rd"].iloc[0], df3_def["Pos"].iloc[0], color='red', s=100)
    ax3.scatter(df3_def["Att 3rd"].iloc[0], df3_def["Pos"].iloc[0], color='red', s=100)
    #increase scale of x axis
    ax1.set_xlim(0, 100)
    ax2.set_xlim(0, 100)
    ax3.set_xlim(0, 100)
    sns.despine(left=True)
    plt.show()
    st.pyplot(fig)

    

    



st.markdown('<h1 style="font-size: 1.5rem;">Shooting </h1>', unsafe_allow_html=True)
#create fourth row
row6_1, row6_2, row6_3 = st.columns(
    (1.6, 1.6, 1.6)
    )
with row6_1:
    #create a column in pl_df called AvgSoT% that is the average of the SoT% columns
    pl_df['AvgSoT%'] = pl_df["SoT%"].mean()
    #pl_df
    #single out player you selected
    pl_df_select = pl_df[pl_df['Player'] == player]
    #pl_df_select
    #create bar chart comparing "SoT%" to "AvgSoT%"
    st.text('Shot on Target %')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3, 3), sharex=True)
    sns.barplot(x="SoT%", y="Player", data=pl_df_select, ax=ax1)
    sns.barplot(x="AvgSoT%", data=pl_df_select, ax=ax2)
    sns.despine(left=True)
    plt.show()
    st.pyplot(fig)
 
#Sh/90
#G/Sh

with row6_2:
    st.text('Shot Value')
    pl_df_pos = pl_df[pl_df['Pos'] == player2['Pos'].iloc[0]]
    #using seaborn create a scatter plot of "G+A" vs "xG+xA"
    #the x axis show "Sh/90"
    #the y axis show "G/Sh"
    #the color shows the player
    fig, (ax1) = plt.subplots(1, 1, figsize=(3, 3), sharex=True)
    sns.scatterplot(x="Sh/90", y="G/Sh", data=pl_df_pos, ax=ax1)
    #highlight the player you selected
    ax1.scatter(pl_df_pos["Sh/90"].iloc[0], pl_df_pos["G/Sh"].iloc[0], color='red', s=100)
    sns.despine(left=True)
    plt.show()
    st.pyplot(fig)

with row6_3:
    st.text('Shot Distance')
    pl_df_pos = pl_df[pl_df['Pos'] == player2['Pos'].iloc[0]]
    #using seaborn create a swarm plot of "Dist"
    fig, (ax1) = plt.subplots(1, 1, figsize=(3, 3), sharex=True)
    ax = sns.swarmplot(x="Dist", y="Pos", data=pl_df_pos, ax=ax1)
    #highlight the player you selected
    ax1.scatter(pl_df_pos["Dist"].iloc[0], pl_df_pos["Pos"].iloc[0], color='red', s=100)
    #increase scale of x axis
    ax1.set_xlim(0, 30)
    sns.despine(left=True)
    ax.set_xlabel('Distance')
    plt.show()
    st.pyplot(fig)


pl_passes_df = pd.read_csv('pl_pass_stats.csv')
pl_passes_type = pd.read_csv('pl_pass_types.csv')

#merge pl_passes_df and pl_df on Rk, player, nation, pos, age, squad, 90s, Born, Att
pl_passes_df = pd.merge(pl_passes_df, pl_passes_type, on=['Rk', 'Player', 'Nation', 'Pos', 'Age', 'Squad', '90s', 'Born', 'Att', 'Cmp', 'Matches'])

#clean pl_passes_df
#in pl_passes_df player column remove anything after "\"
pl_passes_df['Player'] = pl_passes_df['Player'].str.split("\\").str[0]
#in pl_passes_df nation column remove anything before the first capital letter
pl_passes_df['Nation'] = pl_passes_df['Nation'].str.split(" ").str[0]
#make all pl_passes_df nation capitalized
pl_passes_df['Nation'] = pl_passes_df['Nation'].str.upper()

#merge pl_passes_df and pl_df on player, nation, pos, age, squad, 90s, Born, Att, Cmp, Matches
pl_df = pd.merge(pl_df, pl_passes_df, on=['Rk','Player', 'Nation', 'Pos','Squad','Age', 'Born', '90s'])



#in pl_df create a new column called "AvgCmp%.1", "AvgCmp%.2", "AvgCmp%.3"
pl_df['AvgCmp%.1'] = pl_df["Cmp%"].mean()
pl_df['AvgCmp%.2'] = pl_df["Cmp%.1"].mean()
pl_df['AvgCmp%.3'] = pl_df["Cmp%.2"].mean()


 
#using st.bar_chart create a bar chart comparing "Cmp%" to "AvgCmp%"
#st.subheader('Pass Completion %')
#decrease size of subheader
st.markdown('<h1 style="font-size: 1.5rem;">Passing</h1>', unsafe_allow_html=True)

pl_df_select = pl_df[pl_df['Player'] == player]
#in pl_df_select add a new row called AvgCmp%


#pl_df_select create a new dataframe called "pl_df_select_passes" that consists of player, "Cmp%.1", "Cmp%.2", "Cmp%.3", "AvgCmp%.1", "AvgCmp%.2", "AvgCmp%.3"
pl_df_select_passes = pl_df_select[['Player', 'Cmp%', 'Cmp%.1', 'Cmp%.2', 'AvgCmp%.1', 'AvgCmp%.2', 'AvgCmp%.3']]
#create a fifth row
row8_1, row8_spacer1, row8_2, row8_spacer2, row8_3 = st.columns(
    (1.6, .1, 1.6, .1, 1.6)
    )
with row8_1:
    st.text('Pass Completion %')
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(2, 3), sharex=True)
    sns.barplot(x="Cmp%", y="Player", data=pl_df_select_passes, ax=ax1)
    #rename x-axis
    ax1.set_xlabel('Short Pass %')
    sns.barplot(x="AvgCmp%.1", data=pl_df_select_passes, ax=ax2)
    ax2.set_xlabel('Position Average %')
    sns.barplot(x="Cmp%.2", y="Player", data=pl_df_select_passes, ax=ax3)
    ax3.set_xlabel('Long Pass %')
    sns.barplot(x="AvgCmp%.3", data=pl_df_select_passes, ax=ax4)
    ax4.set_xlabel('Position Average %')
    #reduce height of chart
    fig.subplots_adjust(hspace=1.7)
    sns.despine(left=True)
    plt.show()
    st.pyplot(fig)

with row8_2:
    st.text('Pass Variant')
    pl_df_pass = pl_df[pl_df['Pos'] == player2['Pos'].iloc[0]]
    #using seaborn create a normal distrubution chart of "Ground","Low","High"
    fig, (ax1) = plt.subplots(1, 1, figsize=(3, 3), sharex=True)
    ax = sns.distplot(pl_df_pass["Ground"], ax=ax1, kde=False, bins=10, color='red')
    ax = sns.distplot(pl_df_pass["Low"], ax=ax1, kde=False, bins=10)
    ax = sns.distplot(pl_df_pass["High"], ax=ax1, kde=False, bins=10)
    #increase scale of x axis
    ax1.set_xlim(0, 400)
    sns.despine(left=True)
    ax.set_xlabel('Pass Height')
    ax.set_ylabel('Frequency')
    plt.show()
    st.pyplot(fig)
    fig.subplots_adjust(hspace=1.7)

with row8_3:
    #st.subheader('Key Passes')
    st.text('Key Passes')
    #create a beeswarm plot of "KP"
    fig, (ax1) = plt.subplots(1, 1, figsize=(3, 3), sharex=True)
    ax = sns.swarmplot(x="KP", y="Pos", data=pl_df_pass, ax=ax1)
    #highlight the player you selected
    ax1.scatter(pl_df_pass["Dist"].iloc[0], pl_df_pass["Pos"].iloc[0], color='red', s=100)
    #increase scale of x axis
    ax1.set_xlim(0, 100)
    sns.despine(left=True)
    ax.set_xlabel('Key Passes')
    ax.set_ylabel('Pos')
    plt.show()
    st.pyplot(fig)
    #adjust size of graph to be level with other graphs
    fig.subplots_adjust(hspace=1.7)


pl_df_sca = pd.read_csv('pl_sca_gca_stats.csv')

#in pl_df_sca player column remove anything after "\"
pl_df_sca['Player'] = pl_df_sca['Player'].str.split("\\").str[0]
#in pl_df_sca nation column remove anything before the first capital letter
pl_df_sca['Nation'] = pl_df_sca['Nation'].str.split(" ").str[0]
#make all pl_df_sca nation capitalized
pl_df_sca['Nation'] = pl_df_sca['Nation'].str.upper()


st.markdown('<h1 style="font-size: 1.5rem;">Shot and Goal Creation</h1>', unsafe_allow_html=True)
row10_1, row10_spacer1, row10_2, row10_spacer2, row10_3 = st.columns(
    (1.6, .1, 1.6, .1, 1.6)
    )
with row10_1, _lock:
    
    #in pl_df_sca create a new column called "AvgSCA90", "AvgGCA90"
    pl_df_sca['AvgSCA90'] = pl_df_sca["SCA90"].mean()
    pl_df_sca['AvgGCA90'] = pl_df_sca["GCA90"].mean()
    
    #show only selected player
    pl_df_sca_select = pl_df_sca[pl_df_sca['Player'] == player]
    st.text('Per 90')
  
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(3, 3), sharex=True)
    ax = sns.barplot(x="SCA90", y="Player", data=pl_df_sca_select, ax=ax1)
    ax = sns.barplot(x="AvgSCA90", data=pl_df_sca_select, ax=ax2)
    ax = sns.barplot(x="GCA90", y ="Player",data=pl_df_sca_select, ax=ax3)
    ax = sns.barplot(x="AvgGCA90", data=pl_df_sca_select, ax=ax4)
    #increase scale of x axis
    fig.subplots_adjust(hspace=1.7)
    sns.despine(left=True)
    plt.show()
    st.pyplot(fig)

with row10_2, _lock:
    #create a dataframe that shows selected players Pos 
    pl_df_sca_pos = pl_df_sca[pl_df_sca['Pos'] == player2['Pos'].iloc[0]]
    st.text('Shot Creating Ability')
    #create a regplot of "SCAPasslive" and "SCA"
    fig, (ax1) = plt.subplots(1, 1, figsize=(3, 3), sharex=True)
    ax = sns.regplot(x="SCA", y="SCAPassLive", data=pl_df_sca_pos, ax=ax1)
    #highlight the player you selected
    ax1.scatter(pl_df_sca_pos["SCA"].iloc[0], pl_df_sca_pos["SCAPassLive"].iloc[0], color='red', s=100)
   
    #make all other players grey
    
    #increase scale of x axis
    ax1.set_xlim(0, 80)
    ax1.set_ylim(0, 60)
    sns.despine(left=True)
    ax.set_xlabel('SCA')
    ax.set_ylabel('SCA Passes')
    plt.show()
    st.pyplot(fig)


with row10_3, _lock:
    st.text('Goal Creating Ability')
    #create a beeswarm plot of "GCAPassLive"
    pl_df_sca_pos = pl_df_sca[pl_df_sca['Pos'] == player2['Pos'].iloc[0]]
    fig, (ax1) = plt.subplots(1, 1, figsize=(3, 3), sharex=True)
    ax = sns.swarmplot(x="GCAPassLive", y="Pos", data=pl_df_sca_pos, ax=ax1)
    #highlight the player you selected
    ax1.scatter(pl_df_sca_pos["GCAPassLive"].iloc[0], pl_df_sca_pos["Pos"].iloc[0], color='red', s=100)
    #increase scale of x axis
    ax1.set_xlim(0, 15)
    sns.despine(left=True)
    ax.set_xlabel('Goal Creating Passes')
    ax.set_ylabel('Pos')
    plt.show()
    st.pyplot(fig)

