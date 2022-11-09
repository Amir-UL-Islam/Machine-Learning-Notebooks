import pandas as pd
import numpy as np



df_songs = pd.read_feather('data/spotify_2000_2020.feather')
df_songs.head()


samled_df = df_songs.sample(n=1000)
samled_df['duration_minutes'].mean()


df_coffee = pd.read_feather('data/coffee_ratings_full.feather')
df_coffee.head()



