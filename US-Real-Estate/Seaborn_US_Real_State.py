import seaborn as sns 
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 

df = pd.read_csv("RealEstate-USA.csv" , delimiter=",")

print(df)
print(df.info())

# Check Types 
print(df.dtypes)

# describe() methed print count, mean, median, mode and percentile etc
print(df.describe())
print(df.shape)

dfilter = df.head(18)

#  Line Plot
sns.set_theme(style="darkgrid")
sns.lineplot(data = df , x="city" , y= "price")
plt.show()

# categorical plots
sns.set_theme(style="dark")
sns.catplot(data = dfilter , x="city" , y="price")
plt.show

# Plot univariate or bivariate 
sns.set_theme(style="white")
sns.kdeplot(data=dfilter , x="zip_code" , y="price")
plt.show


# Scatter plot
sns.set_theme(style="white")
sns.scatterplot(data=dfilter ,x="zip_code" , y="price")
plt.show

# Barplot
sns.set_theme(style="dark")
sns.barplot(data=dfilter , x="zip_code" , y="price")
plt.show

# Heatmap 
glue = dfilter.pivot(columns="zip_code" , values="price")
sns.set_theme(style="darkgrid")
sns.heatmap(data=glue)
plt.show

# Lineplot
sns.set_theme(style="whitegrid")
sns.lineplot(data=dfilter , x="zip_code" , y="price")
plt.show()

# Catplot
sns.set_theme(style="whitegrid")
sns.catplot(data=dfilter , x="zip_code" , y="price")
plt.show()

# Plot univariate or bivariate (kdeplot)
sns.set_theme(style="white")
sns.kdeplot(data=dfilter , x="zip_code" , y="price")
plt.show()

