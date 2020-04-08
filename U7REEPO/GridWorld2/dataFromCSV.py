import pandas as pd
import plotly.express as px

df = pd.read_csv('tensorBoard_Pong_GW/progress.csv')

fig = px.line(df, x = 'steps', y = 'mean 100 episode reward', title='Mean 100 episode reward')
fig.show()