#!/usr/bin/env python
# coding: utf-8

# ### <span style="color:blue"> Task 1: Data Exploration </span>
# 
# #### <span style="color:indigo">Sub Task 1: Initial Observation </span>
# 
# 

# In[1]:


# As the first step towards analysis, I have installed required libraries
get_ipython().system('pip install numpy')
get_ipython().system('pip install pandas')
get_ipython().system('pip install matplotlib')
get_ipython().system('pip install seaborn')


# In[2]:


import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


#Let's read the dataset an assign it as a dataframe
netflix_df= pd.read_csv("https://d2beiqkhq929f0.cloudfront.net/public_assets/assets/000/000/940/original/netflix.csv")


# In[4]:


#Let's check first 10 rows of dataframe
netflix_df.head(10)


# In[5]:


#Let's check the shape of the dataframe
netflix_df.shape


# In[6]:


#Let's check total no. of rows, columns and data type of different variables
netflix_df.info()


# In[7]:


# Let's see unique attributes (columns) in the DataFrame
unique_attributes = netflix_df.columns.tolist()

# Print unique attributes
print(unique_attributes)


# In[8]:


#Let's check all the categorical variables
data_types = netflix_df.dtypes
categorical_attributes = data_types[data_types == 'object'].index.tolist()
print("Categorical Attributes:")
print(categorical_attributes)


# In[9]:


#Let's check the no. of unique values in our dataframe columnwise
for i in netflix_df.columns:
    print(i, ':',netflix_df[i].nunique())


# In[10]:


# Let's check the occurences of each of the ids
netflix_df['show_id'].value_counts()


# In[11]:


# Let's check the occurences of each of the values in type
netflix_df['type'].value_counts()


# In[12]:


# Let's check the occurences of each of the titles
netflix_df['title'].value_counts()


# In[13]:


# Let's check the occurences of each of the director
netflix_df['director'].value_counts()


# In[14]:


# Let's check the occurences of each of the cast
netflix_df['cast'].value_counts()


# In[15]:


# Let's check the occurences of each of the country
netflix_df['country'].value_counts()


# In[16]:


# Let's check the occurences of each of the date added in Netflix
netflix_df['date_added'].value_counts()


# In[17]:


# Let's check the occurences of each of the release year
netflix_df['release_year'].value_counts()


# In[18]:


# Let's check the occurences of each of the rating
netflix_df['rating'].value_counts()


# In[19]:


# Let's check the occurences of each of the duration of Movies/TV shows
netflix_df['duration'].value_counts()


# In[20]:


# Let's check the occurences of each of the contents listed in various categories
netflix_df['listed_in'].value_counts()


# In[21]:


#Let's check the null values in our dataframe 

print(netflix_df.isnull().sum())


# In[22]:


#let's count the total no. of null values across dataframe
print(netflix_df.isnull().sum().sum())


# In[23]:


#Let's check the presence of duplicate values as well
print(netflix_df.duplicated().sum())


# #### After initial observation, there are total <span style="color:blue">8807 no. of rows(observations) </span> and <span style="color:blue">12 no. of columns(variables)</span> . Out of 12 no. of variables, only <span style="color:blue">release_year </span>  is particularly of integer type but rest are of object type. The columns, 'director',  'cast', 'country', 'date_added', 'rating', and 'duration' have null values, comprising 4307 total null values across the dataframe. However, there are no duplicate values.

# #### <span style="color:indigo">Sub Task 2: Data Preprocessing </span>

# In[24]:


#Let's impute the null/missing values 
# The 'director', 'cast' and 'country' columns have high null values and deleting them will affect the dataframe 
# due to information loss. So, we will use fillna()approach 

netflix_df.director.fillna("Director Unavailable", inplace=True)
netflix_df.cast.fillna("Cast Unavailable", inplace=True)
netflix_df.country.fillna("Country Unavailable", inplace=True)


# In[25]:


# The columns 'date_added', 'rating' and 'duration' have very few null values so I have deleted those
netflix_df.dropna(subset=["date_added", "rating", "duration"], inplace=True)


# In[26]:


#let's check the output for null values imputation
netflix_df.isnull().any()


# In[27]:


#let's check summary statistics of our dataframe
summary_stat = netflix_df.describe()
summary_stat


# In[28]:


#based on above summary, there seems to be no outliers. However, let's use IQR method for more precise output

# let's calculate the first quartile (Q1)
Q1= netflix_df['release_year'].quantile(0.25)
# let's calculate the first quartile (Q3)
Q3= netflix_df['release_year'].quantile(0.75)

IQR = Q3 - Q1

#Let's identify outliers
# let the threshold be 1.5
threshold= 1.5

# let's define lower and upper bounds
lower_limit = Q1- 1.5* IQR
upper_limit = Q3+ 1.5* IQR

release_year_outliers = netflix_df[(netflix_df['release_year'] < lower_limit)
                                   | (netflix_df['release_year'] > upper_limit)]

# Print potential outliers in the release_year column
print("Potential outliers in release_year:")
print(release_year_outliers['release_year'])


# In[29]:


# There are 717 observations representing as outliers so instead of deleting we will replace with median values

median_release_year = netflix_df['release_year'].median()

netflix_df.loc[(netflix_df['release_year'] < lower_limit) 
               | (netflix_df['release_year'] > upper_limit), 'release_year'] = median_release_year


# In[30]:


# Let's check if the operation was successful
summary_stat_after_replacement = netflix_df.describe()

# Print summary statistics after median replacement
print(summary_stat_after_replacement)


# In[31]:


netflix_df


# #### <span style="color:orange">The columns like 'director', 'cast', 'country', 'listed_in' contain hierarchial data. So, for ease of analysis let's unnest these columns one by one. </span> 

# In[32]:


#unnesting director column
constraint1 = netflix_df['director'].apply(lambda x: str(x).split(',')).tolist()

netflix_df_new1 = pd.DataFrame(constraint1, index=netflix_df['title'])
netflix_df_new1 = netflix_df_new1.stack()
netflix_df_new1 = pd.DataFrame(netflix_df_new1.reset_index())
netflix_df_new1.rename(columns={0: "Directors"}, inplace=True)
netflix_df_new1.drop(["level_1"], axis=1, inplace=True)
netflix_df_new1.head()



# In[33]:


# Unnesting the cast column, i.e. creating separate lines for each cast member in a movie
constraint2 = netflix_df['cast'].apply(lambda x: str(x).split(', ')).tolist()

netflix_df_new2 = pd.DataFrame(constraint2, index=netflix_df['title'])
netflix_df_new2 = netflix_df_new2.stack()
netflix_df_new2 = pd.DataFrame(netflix_df_new2.reset_index())
netflix_df_new2.rename(columns={0: "Actors"}, inplace=True)
netflix_df_new2.drop(["level_1"], axis=1, inplace=True)
netflix_df_new2.head()


# In[34]:


# Unnesting the cast column, i.e. creating separate lines for each country for a movie
constraint3 = netflix_df['country'].apply(lambda x: str(x).split(', ')).tolist()

netflix_df_new3 = pd.DataFrame(constraint3, index=netflix_df['title'])
netflix_df_new3 = netflix_df_new3.stack()
netflix_df_new3 = pd.DataFrame(netflix_df_new3.reset_index())
netflix_df_new3.rename(columns={0: "Country"}, inplace=True)
netflix_df_new3.drop(["level_1"], axis=1, inplace=True)
netflix_df_new3.head()


# In[35]:


# Unnesting the listed_in column, i.e. creating separate lines for each genre in a movie
constraint4 = netflix_df['listed_in'].apply(lambda x: str(x).split(', ')).tolist()

netflix_df_new4 = pd.DataFrame(constraint4, index=netflix_df['title'])
netflix_df_new4 = netflix_df_new4.stack()
netflix_df_new4 = pd.DataFrame(netflix_df_new4.reset_index())
netflix_df_new4.rename(columns={0: "Genre"}, inplace=True)
netflix_df_new4.drop(["level_1"], axis=1, inplace=True)
netflix_df_new4.head()


# In[36]:


#Let's Merge our unnested data with original data

desired_columns = ['show_id', 'type', 'title', 'Directors',
                   'Actors', 'Country', 'Genre', 'date_added', 'release_year', 'rating', 'duration']

# Merge each new DataFrame on the right side
netflix_df_final = netflix_df[['show_id', 'type', 'title',
                               'date_added', 'release_year', 'rating', 'duration']].copy()
netflix_df_final = netflix_df_final.merge(netflix_df_new1, on='title', how='left')
netflix_df_final = netflix_df_final.merge(netflix_df_new2, on='title', how='left')
netflix_df_final = netflix_df_final.merge(netflix_df_new3, on='title', how='left')
netflix_df_final = netflix_df_final.merge(netflix_df_new4, on='title', how='left')

# Reorder the columns
netflix_df_final = netflix_df_final[desired_columns]
netflix_df_final.head()


# In[37]:


#Now let's check if there is any null values in our final merged data

netflix_df_final.isnull().sum()


# In[38]:


#Let's check the duration column
netflix_df_final['duration'].value_counts()


# In[39]:


#Let's remove min from movie duration values
netflix_df_final['duration']=netflix_df_final['duration'].str.replace("min","")
netflix_df_final.head()


# In[40]:


# Let's check unique values in duration
netflix_df_final['duration'].unique()


# In[41]:


# Let's create a new column and remove the seasons from duration column as it will create confusion during visual analyis
# and will keep it in a new dataframe
netflix_df_final['duration_copy']=netflix_df_final['duration'].copy()
netflix_df_final1= netflix_df_final.copy()
netflix_df_final1.loc[netflix_df_final1['duration_copy'].str.contains('Season'), 'duration_copy']=0
netflix_df_final1['duration_copy']=netflix_df_final1['duration_copy'].astype('int')
netflix_df_final1.head()


# In[42]:


netflix_df_final1['duration_copy'].describe()


# ### <span style="color:blue"> Task 2: Data Visualization </span>

# In[43]:


#Let's visualise the total no. of movies and TV shows to check the majority
color = ['Violet', 'Green']
# Plot the count plot
type_counts = netflix_df_final['type'].value_counts()

# Plot the value counts
ax = sns.barplot(x=type_counts.index, y=type_counts.values, palette=color)

# Add counts above the bars
for i, count in enumerate(type_counts.values):
    ax.text(i, count + 0.1, str(count), ha='center')

plt.title('Count of Different shows on Netflix')
plt.xlabel('Type')
plt.ylabel('Count')
plt.show()


# #### There are 145831 Movies and 55932 TV Shows.

# In[44]:


#Let's visualise the total no. of movies and TV shows to check the majority
color = ['Violet', 'Green']

plt.figure(figsize=(8, 6))
sns.boxplot(x='type', y='release_year', data=netflix_df_final, palette=color)

# Set title and labels
plt.title('Distribution of Release Year by Content Type')
plt.xlabel('Content Type')
plt.ylabel('Release Year')

# Show the plot
plt.show()


# #### Newer releases are more common on Netflix, with a median release year of 2017 for movies and 2018 for TV shows.
# #### There is a wider range of release years for movies compared to TV shows.
# #### There are a few outliers for both movies and TV shows, suggesting a small number of titles with release years much earlier or later than the majority.

# In[45]:


#Let's see the movie duration distribution

sns.distplot(netflix_df_final1['duration_copy'], hist= True, 
             kde= True, bins= int(36), color= 'green',
            hist_kws= {'edgecolor':'black'},
            kde_kws= {'linewidth':4})
plt.show()


# #### Majority of the movies are less than 60 minutes and its being followed by the movies having 100 minutes of duration.

# In[46]:


#Let's check the top 10 countries with highest content on Netflix
# Ihave filtered the countries so as to not include Country Unavailable ones
# Split countries and stack them
filtered_countries = netflix_df_final.set_index('title')['Country'].str.split(', ', expand=True).stack()

# Drop 'Country Unavailable'
filtered_countries = filtered_countries[filtered_countries != 'Country Unavailable']

# Plot
plt.figure(figsize=(13,7))
g = sns.countplot(y=filtered_countries, order=filtered_countries.value_counts().index[:10])
plt.title('Top 10 Countries Contributor on Netflix')
plt.xlabel('Titles')
plt.ylabel('Country')
plt.show()


# #### United States is the top contributor in terms of contents available on Netflix.

# In[47]:


# Let's check who are the top directors of contents on Netflix
filtered_directors = netflix_df_final[netflix_df_final.Directors != 'Director Unavailable'].set_index('title').Directors.str.split(
    ', ', expand=True).stack().reset_index(level=1, drop=True)

plt.figure(figsize=(13,7))
plt.title('Top 10 Director Based on The Number of Titles')
sns.countplot(y = filtered_directors, order=filtered_directors.value_counts().index[:10], palette='hls')
plt.show()


# #### Martin Scorsese is the top director on Netflix.

# In[48]:


# Let's check top Genre on Netflix

filtered_genres = netflix_df_final.set_index('title').Genre.str.split(
    ', ', expand=True).stack().reset_index(level=1, drop=True);

plt.figure(figsize=(10,7))
g = sns.countplot(y = filtered_genres, order=filtered_genres.value_counts().index[:10])
plt.title('Top 20 Genres on Netflix')
plt.xlabel('Titles')
plt.ylabel('Genres')
plt.show()


# #### Dramas and International Movies are the top Genres on Netflix.

# In[49]:


# Let's check the top 10 actors on Netflix based on no. of titles

netflix_tvshows_df = netflix_df_final[netflix_df_final.type.str.contains("TV Show")]
filtered_cast_shows = netflix_tvshows_df[netflix_tvshows_df.Actors != 'Cast Unavailable'].set_index('title').Actors.str.split(
    ', ', expand=True).stack().reset_index(level=1, drop=True)

plt.figure(figsize=(13,7))
plt.title('Top 10 Actor TV Show Based on The Number of Titles')
sns.countplot(y = filtered_cast_shows, order=filtered_cast_shows.value_counts().index[:10], palette='Paired')
plt.show()


# In[50]:


#Let's check the top actors on Netflix movies based on the content

netflix_movies_df = netflix_df_final[netflix_df_final.type.str.contains("Movie")]
filtered_cast_movie = netflix_movies_df[netflix_movies_df.Actors != 'Cast Unavailable'].set_index('title').Actors.str.split(
    ', ', expand=True).stack().reset_index(level=1, drop=True)

plt.figure(figsize=(13,7))
plt.title('Top 10 Actor Movies Based on The Number of Titles')
sns.countplot(y = filtered_cast_movie, order=filtered_cast_movie.value_counts().index[:10], palette='pastel')
plt.show()


# In[51]:


# Let's check the amount of contents by ratings
plt.figure(figsize=(10, 6))
sns.countplot(data=netflix_df_final, x='rating', hue='type', palette= "tab10")

# Set the title and labels
plt.title('Amount of Content by Ratings')
plt.xlabel('Rating')
plt.ylabel('Count')

# Show the plot
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.legend(title='Type')
plt.tight_layout()  # Adjust layout to prevent clipping of labels
plt.show()


# In[52]:


# Let's set up the figure with joint and marginal histograms
g = sns.JointGrid(data=netflix_df_final, x='release_year', y='Genre', height=8)

# Let's plot the scatterplot on the joint axes
g.plot_joint(sns.scatterplot, hue='type', data=netflix_df_final)

# Plot the histograms on the marginal axes
sns.histplot(data=netflix_df_final, x='release_year', ax=g.ax_marg_x, bins=30, kde=True)
sns.histplot(data=netflix_df_final, y='Genre', ax=g.ax_marg_y, bins=30, kde=True)

# Set titles and labels
g.set_axis_labels('Release Year', 'Genre')
g.fig.suptitle('Jointplot of Release Year vs Genre', y=1.02)

# Show the plot
plt.show()
