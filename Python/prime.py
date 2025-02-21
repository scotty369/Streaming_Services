#%%
import polars as pl
import pandas as pd
import plotly_express as px
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

df = pl.read_csv("/Users/scotttow123/Documents/Streaming_Services/Data/amazon_prime_titles.csv")

print(df.head)

print(df.schema)

missing_values = df.null_count()
print(missing_values)

print(df.describe)
#%%
import polars as pl
import plotly.express as px

genres = df.select("listed_in").to_series().str.split(", ").explode()
genre_counts = genres.value_counts()  

genre_counts.columns = ["Genre", "Count"]

fig = px.bar(genre_counts.to_pandas(), x="Genre", y="Count", 
             labels={"Genre": "Genre", "Count": "Count"},
             title="Most Common Genres on Netflix")
fig.show()
#%%
import seaborn as sns
import matplotlib.pyplot as plt

movies = df.filter(df["type"] == "Movie")

movies = movies.with_columns(
    movies["duration"]
    .str.extract(r"(\d+)", 0)
    .cast(pl.Int32, strict=False)
    .alias("duration_minutes")
)

sns.boxplot(data=movies.to_pandas(), x="rating", y="duration_minutes")
plt.title("Movie Duration by Rating")
plt.xticks(rotation=45)
plt.show()
# %%
import numpy as np

durations = movies["duration_minutes"].drop_nulls().to_numpy()

average_duration = np.mean(durations)
print(f"Average Movie Duration: {average_duration} minutes")
# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("/Users/scotttow123/Documents/Streaming_Services/Data/netflix_titles.csv")

movies = df[df['type'] == 'Movie'].dropna(subset=['duration'])

movies['duration_minutes'] = movies['duration'].str.extract('(\d+)').astype(float)

plt.figure(figsize=(10, 6))
sns.histplot(movies['duration_minutes'], kde=True, bins=30, color='blue')
plt.title("Distribution of Movie Durations on Netflix")
plt.xlabel("Duration (Minutes)")
plt.ylabel("Frequency")
plt.show()
# %%
import plotly.express as px

fig = px.scatter(movies, x='rating', y='duration_minutes', 
                 title="Movie Duration vs. Rating", 
                 labels={"rating": "Rating", "duration_minutes": "Duration (Minutes)"})
fig.show()
# %%
directors = df[df['type'] == 'Movie']['director'].dropna().str.split(',').explode()

director_counts = directors.value_counts().head(10).reset_index()
director_counts.columns = ['Director', 'Count']

fig = px.bar(director_counts, x='Director', y='Count',
             title="Top 10 Directors with Most Movies on Netflix",
             labels={'Director': 'Director', 'Count': 'Count'})
fig.show()
# %%
avg_duration = movies.groupby('rating')['duration_minutes'].mean().reset_index()

plt.figure(figsize=(10, 6))
sns.barplot(data=avg_duration, x='rating', y='duration_minutes', palette='viridis')
plt.title("Average Movie Duration by Rating")
plt.xlabel("Rating")
plt.ylabel("Average Duration Minutes")
plt.xticks(rotation=45)
plt.show()
# %%
plt.figure(figsize=(10, 6))
sns.scatterplot(data=movies, x='release_year', y="duration_minutes", hue='rating', palette='coolwarm', alpha=0.6)
plt.title=("Release Year vs Movie Duration")
plt.xlabel("Release Year")
plt.ylabel("Duration (Minutes)")
plt.show()
# %%
import pandas as pd
import plotly.express as px

countries = df['country'].dropna().str.split(',').explode()

country_counts = countries.value_counts().head(10).reset_index()
country_counts.columns = ['Country', 'Count']

fig = px.bar(country_counts, x='Country', y='Count',
            title='Top 10 Countries Producing Netflix Content',
            labels={'Country': 'Country', 'Count': 'Count'})
fig.show()
# %%
