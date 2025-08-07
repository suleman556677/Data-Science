import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""Built-in Themes
Seaborn provides five built-in themes:
darkgrid: Adds a gray background with white gridlines. It is the default theme.
whitegrid: Adds gray gridlines on a white background.
dark: Similar to darkgrid but without the gridlines.
white: Similar to whitegrid but without the gridlines.
ticks: Adds ticks to the axes and uses a white background.
Setting Themes
The seaborn.set_theme() or seaborn.set_style() function can be used to set the theme for all plots. """


# Fast Food Restaurant Data
# Load Data From A CSV File 
df = pd.read_csv("FastFoodRestaurants (4).csv", delimiter=",")
print("Data - Frame ", df)

df_filter = df.head(10)

# Set The Theme
sns.set_theme(style="dark")

# Create A LinePlot 
sns.lineplot(x="city", y="name", data=df_filter)
plt.show()

# Other Theme Can Be Set Similarly
sns.set_theme(style="darkgrid")
sns.lineplot(x="city", y="name", data=df_filter)
plt.show()

# Set The Theme (white)
sns.set_theme(style="white")
sns.lineplot(x="city", y="name", data=df_filter)
plt.show()

# Set The Theme (whitegrid)
sns.set_theme(style="whitegrid")
sns.lineplot(x="city", y="name", data=df_filter)
plt.show()

# Set The Theme (ticks)
sns.set_theme(style="ticks")
sns.lineplot(x="city", y="name", data=df_filter)
plt.show()

# Customize The Themes 
sns.set_theme(style="darkgrid", rc={"axes.facecolor" : "grey", "grid.color" : "white"})
sns.lineplot(x="city", y="name", data=df_filter)
plt.show()

"""seaborn.set_theme() allows customization of the appearance of plots by modifying matplotlib's rc parameters. It accepts a dictionary rc to override default settings. Here's a breakdown of commonly used rc parameters:
axes.facecolor: Background color of the plotting area (e.g., 'white', '#EAEAF2').
axes.edgecolor: Color of the axes lines (e.g., 'black', 'gray').
axes.linewidth: Width of the axes lines in points.
axes.grid: Whether to show the grid ('True' or 'False').
axes.grid.axis: Which axes to show the grid lines on ('x', 'y', or 'both').
axes.grid.which: Which grid lines to draw ('major', 'minor', or 'both').
axes.labelcolor: Color of the axis labels.
axes.labelsize: Size of the axis labels in points or as a relative string (e.g., 'large', 'small').
axes.titlesize: Size of the plot title.
xtick.color: Color of the x-axis tick marks and labels.
ytick.color: Color of the y-axis tick marks and labels.
xtick.labelsize: Size of the x-axis tick labels.
ytick.labelsize: Size of the y-axis tick labels.
grid.color: Color of the grid lines.
grid.linewidth: Width of the grid lines.
font.family: Font family to use (e.g., 'sans-serif', 'serif', 'monospace').
font.size: Default font size for text elements.
lines.linewidth: Width of lines in plots.
lines.linestyle: Style of lines (e.g., '-', '--', '-.', ':').
patch.edgecolor: Color of patch edges (e.g., in histograms, bar plots).
patch.linewidth: Width of patch edges.
legend.frameon: Whether to display a frame around the legend ('True' or 'False').
legend.fontsize: Size of the legend text.
figure.figsize: Size of the figure (width, height) in inches.
figure.facecolor: Background color of the entire figure."""

# Set Theme 
sns.set(style="whitegrid")

# kind="hist" 
sns.displot(data=df_filter, x="latitude", y="longitude", hue="name", kind="hist")

# Show The Plot
plt.show()

""""kind="kde" in Seaborn specifies the use of Kernel Density Estimation plots. KDE plots visualize the probability density of a continuous variable. Instead of discrete bins like in histograms, KDE plots use a continuous curve to estimate the underlying distribution of the data. This provides a smoother and often more informative representation of the data's distribution, especially for continuous variables."""
#kind='kde'
sns.displot(data=df_filter, x="latitude", y="longitude", hue="province")

# Display the plot
plt.show()

#kind='kde'
sns.kdeplot(data=df_filter, x="latitude")
plt.suptitle("Kdeplot")

# Display the plot
plt.show()

"""Plot univariate or bivariate histograms to show distributions of datasets.
A histogram is a classic visualization tool that represents the distribution 
of one or more variables by counting the number of observations that fall 
within discrete bins."""

sns.histplot(data=df_filter, x="city", y="postalCode", hue="city", multiple="stack")

# Display the plot
plt.show()

"""Draw a scatter plot with possibility of several semantic groupings.
The relationship between x and y can be shown for different subsets of
the data using the hue, size, and style parameters. These parameters 
control what visual semantics are used to identify the different subsets.
It is possible to show up to three dimensions independently by using all three semantic types, 
but this style of plot can be hard to interpret and is often ineffective. 
Using redundant semantics (i.e. both hue and style for the same variable) 
can be helpful for making graphics more accessible."""

# Use Seaborn To Create A Plot
# ScatterPlot
sns.scatterplot(data=df_filter, x="latitude", y="longitude")
plt.show()

"""Draw a line plot with possibility of several semantic groupings.
The relationship between x and y can be shown for different subsets
of the data using the hue, size, and style parameters. 
These parameters control what visual semantics are 
used to identify the different subsets. It is possible 
to show up to three dimensions independently by using 
all three semantic types, but this style of plot can 
be hard to interpret and is often ineffective. 
Using redundant semantics (i.e. both hue and style 
for the same variable) can be helpful for making graphics more accessible."""

# LinePlot
sns.lineplot(data=df_filter, x="latitude", y="longitude")

# Display the plot
plt.show()

"""Show point estimates and errors as rectangular bars.
A bar plot represents an aggregate or statistical estimate 
for a numeric variable with the height of each rectangle 
and indicates the uncertainty around that estimate using 
an error bar. Bar plots include 0 in the axis range, 
and they are a good choice when 0 is a meaningful value 
for the variable to take."""

# BarPlot
sns.barplot(data=df_filter, x="latitude", y="longitude")

# Display the plot
plt.show()


""""Figure-level interface for drawing categorical plots onto a FacetGrid.
This function provides access to several axes-level functions that show the 
relationship between a numerical and one or more categorical variables using 
one of several visual representations. The kind parameter selects the underlying 
axes-level function to use."""

# CatPlot
sns.catplot(data=df_filter, x="latitude", y="longitude")
plt.show()

""""Plot rectangular data as a color-encoded matrix.
This is an Axes-level function and will draw the heatmap 
into the currently-active Axes if none is provided to the 
ax argument. Part of this Axes space will be taken and used 
to plot a colormap, unless cbar is False or a separate Axes 
is provided to cbar_ax."""

glue = df_filter.pivot(columns="latitude", values="longitude")
sns.heatmap(glue)
plt.show()


""""Seaborn provides a variety of plot types for different data visualization needs. These can be broadly categorized as: 
Relational plots: 

    These plots visualize the relationship between two or more variables. 

    scatterplot(): Displays the relationship between two numerical variables using points. 

    lineplot(): Shows the relationship between two numerical variables as a line, often used for time series data or trends. 

Categorical plots: 

    These plots display the distribution of a variable across different categories. 

    barplot(): Shows the mean of a numerical variable for different categories, with error bars. 

    countplot(): Displays the frequency of each category. 

    boxplot(): Shows the distribution of a numerical variable for different categories using quartiles. 

    violinplot(): Similar to a boxplot, but also shows the probability density of the data. 

    stripplot(): Displays individual data points for each category, allowing for visualization of the distribution. 

    swarmplot(): Similar to a stripplot, but the points are adjusted to avoid overlapping, providing a better view of the distribution. 

Distribution plots: 

    These plots visualize the distribution of a single variable. 

    histplot(): Displays the frequency distribution of a numerical variable using bins. 

    kdeplot(): Shows the estimated probability density function of a numerical variable. 

    ecdfplot(): Displays the empirical cumulative distribution function of a numerical variable. 

Multi-plot grids: 

    These functions allow for creating multiple plots at once, useful for comparing relationships between multiple variables. 

    pairplot(): Creates a matrix of scatter plots for all pairs of variables in a dataset. 

    relplot(): Facets scatterplot() and lineplot() across additional categorical variables. 

    catplot(): Facets categorical plots across additional categorical variables. 

Regression plots: 

    These plots visualize the relationship between two variables and fit a regression model. 

    lmplot(): Combines regplot() with FacetGrid to show linear relationships with the option to condition on other variables. 

    regplot(): Shows the relationship between two variables with a fitted regression line. 

Other plots: 

    heatmap(): Displays a matrix of values as a heatmap, often used for correlation matrices or other tabular data. 


Note: The choice of plot type depends on the type of data and the insights you want to extract. For comparing distributions, boxplot() or violinplot() are suitable. For visualizing relationships between two variables, scatterplot() or lineplot() can be used. For exploring relationships between multiple variables, pairplot() is a powerful tool. 

 """



















