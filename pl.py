
#import basic visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
#clean columns

from google.colab import drive
drive.mount('/content/drive')

!ls drive/'My Drive'/ML_Process_Data_Files/Section_6_EDA

#read in data
df_agg = pd.read_csv('drive/My Drive/ML_Process_Data_Files/Section_6_EDA/Aggregated_Metrics_By_Video.csv',encoding='utf-8')
df_agg_country_sub = pd.read_csv('drive/My Drive/ML_Process_Data_Files/Section_6_EDA/Aggregated_Metrics_By_Country_And_Subscriber_Status.csv', encoding='utf-8')
df_ts = pd.read_csv('drive/My Drive/ML_Process_Data_Files/Section_6_EDA/Video_Performance_Over_Time.csv', encoding='utf-8')
df_comments = pd.read_csv('drive/My Drive/ML_Process_Data_Files/Section_6_EDA/All_Comments_Final.csv', encoding='utf-8')

#look at columns for each dataframe
print(df_agg.columns)
print(df_agg_country_sub.columns)
print(df_ts.columns)
print(df_comments.columns)

#the column headers have some extra non-ascii characters, we need to clean them up before we do our analysis
#this goes through each column and removes all the non-ascii characters

newcols =[x.encode("ascii", "ignore").decode('utf-8') for x in df_agg.columns]
df_agg.columns = newcols

df_agg.columns

#Using .describe() gives us basic descriptive statistics for all of our numeric data. From this we can see which variables we might want to explore more.
#We can see things like: which variables might have outliers, which might have skew, or which have a wide range of values
df_agg.describe()

"""## Single variable plots
After looking at the descriptive statistics, we may want to explore our numeric features more. To do this, we like to use single variable plots. These can help us understand the distributions of our data. If our data doesn't follow a normal distribution, we may want to make some transform it so it can be used by specific types of models like linear regresison.

Looking at these charts can also help us to evaluate if there are outliers present in our data.

First, we will look at histograms of our data. These help us to see skew as well as some outliers. I usually do this with every feature (if possible). Try more of these on your own!

Next, we will explore some box plots to see if they tell us additional information

Finally, we will explore some distribution plots for categorical variables.
"""

#you can easily see the distribuiton of data in a column with panda's built in .hist() method. Here is the distribution for likes. There is clearly one sample that has way more like sthan the others, this is something we could look into more

df_agg.Likes.hist(bins = 100)

#let's look at the same thing for average percentage viewed. This will have no outliers unlike the likes column. We are using matplotlib's hist function here instead of the integration through pandas.
plt.hist(df_agg['Average percentage viewed (%)'])

#this data has a significant amount of right skew, we may want to transform this data if we were planning to use linear regression.
df_agg['Impressions click-through rate (%)'].hist()

#From this boxplot, we can see that there are quite a few outliers from our normal data. What does this tell us about the nature of likes on videos? Perhaps we shouldn't be trusting averages if something can skew so high.
plt.boxplot(df_agg['Likes'])

#this plot looks a lot more normal. We can se emost of the videos are viewed between 25-45% of the way through. We still have some outliers, what may be special or different about those?
plt.boxplot(df_agg['Average percentage viewed (%)'])
#

#with click through rate, we also see quite a few high outliers versus the median of / interquartile range.
plt.boxplot(df_agg['Impressions click-through rate (%)'])

#To show how to do this with categorical data, let's make a categorical column.
#We can take our revenue data and make it into different categories that are relevant to me. Usually I'm interested if videos have made less than $100,
#between $100-1000 and over $1000.

#Let's quickly "engineer these categories for ourself"

#make bins from 0-100, 100-1000, and greater than 1000
bins = pd.IntervalIndex.from_tuples([(0, 100), (100, 1000), (1000,float("inf"))])
df_agg['rev_buckets'] = pd.cut(df_agg['Your estimated revenue (USD)'],bins)

#get count of number of videos by reveune bucket
rev_values = df_agg['rev_buckets'].value_counts()
rev_values.plot.bar()