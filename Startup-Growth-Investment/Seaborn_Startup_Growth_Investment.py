import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 

# Load CSV File 
df = pd.read_csv("startup_growth_investment_data (2).csv", delimiter=",")
print("DataFrame", df)
print(df.dtypes)
df_filter = df.head(20)

# Set The Theme (darkgrid)
sns.set_theme(style="darkgrid")

# Create A LinePlot
sns.lineplot(data=df, x="Funding Rounds", y="Growth Rate (%)")

# Display Plot
plt.show()

# Other themes can be set similarly
# Set The Theme (dark)
sns.set_theme(style="dark") 

# Create A LinePlot
sns.lineplot(data=df, x="Funding Rounds", y="Growth Rate (%)")

# Display Plot
plt.show()

# Set The Theme (white)
sns.set_theme(style="white") 

# Create A LinePlot
sns.lineplot(data=df, x="Funding Rounds", y="Growth Rate (%)")

# Display Plot
plt.show()

# Set The Theme (whitegrid)
sns.set_theme(style="whitegrid") 

# Create A LinePlot
sns.lineplot(data=df, x="Funding Rounds", y="Growth Rate (%)")

# Display Plot
plt.show()

# Set The Theme (ticks)
sns.set_theme(style="ticks") 

# Create A LinePlot
sns.lineplot(data=df, x="Funding Rounds", y="Growth Rate (%)")

# Display Plot
plt.show()

"""Customizing Themes
It is possible to customize the themes further by passing a dictionary of parameters to the rc argument of seaborn.set_theme() or seaborn.set_style(). This allows for fine-grained control over the appearance of plots."""

# Customize the theme
sns.set_theme(style="darkgrid", rc={"axes.facecolor": "grey", "grid.color": "white"})

# Create A LinePlot
sns.lineplot(data=df, x="Funding Rounds", y="Growth Rate (%)")

# Display Plot
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
legend.'frameon: Whether to display a frame around the legend ('True' or 'False').
legend.fontsize: Size of the legend text.
figure.figsize: Size of the figure (width, height) in inches.
figure.facecolor: Background color of the entire figure."""

# Set Theme (whitegrid)
sns.set_theme(style="whitegrid")

# Kind="hist"
# displot    -  Figure-level Interface For Drawing Distribution Plots Onto A FacetGrid.
sns.displot(data=df_filter, x="Funding Rounds", y="Valuation (USD)", hue="Funding Rounds", kind="hist")
plt.suptitle("Funding_Round And Valuation_(USD)")
plt.show()

"""kind="kde" in Seaborn specifies the use of Kernel Density Estimation plots. KDE plots visualize the probability density of a continuous variable. Instead of discrete bins like in histograms, KDE plots use a continuous curve to estimate the underlying distribution of the data. This provides a smoother and often more informative representation of the data's distribution, especially for continuous variables."""
# Kind="kde"
# displot    -  Figure-level Interface For Drawing Distribution Plots Onto A FacetGrid.
sns.set_theme(style="darkgrid")
sns.displot(data=df_filter, x="Number of Investors", hue="Growth Rate (%)", kind="kde")
plt.suptitle("Distribution Of Growth Rate (%)")
plt.show()

# kdeplot    -  Plot Univariate Or Bivariate Distributions Using Kernel Density Estimation.

# Kind="kde"
sns.set_theme(style="dark")
sns.kdeplot(data=df_filter, x="Growth Rate (%)" )
plt.suptitle("Growth Rate (%)")
plt.show()

# histplot   -  Plot Univariate Or Bivariate Histograms To Show Distributions Of Datasets.
sns.set_theme(style="whitegrid")
sns.histplot(data=df_filter, x="Number of Investors", bins=10, kde=True)
plt.suptitle("Number Of Investors Distribution")
plt.show()

"""Draw a scatter plot with possibility of several semantic groupings.

The relationship between x and y can be shown for different subsets of the data using the hue, size, and style parameters. These parameters control what visual semantics are used to identify the different subsets. It is possible to show up to three dimensions independently by using all three semantic types, but this style of plot can be hard to interpret and is often ineffective. Using redundant semantics (i.e. both hue and style for the same variable) can be helpful for making graphics more accessible."""

# ScatterPlot
sns.set_theme(style="whitegrid")
sns.scatterplot(data=df_filter, x="Number of Investors", y="Valuation (USD)")
plt.suptitle("Investment vs Valuation by Industry")
plt.show()

"""Draw a line plot with possibility of several semantic groupings.

The relationship between x and y can be shown for different subsets of the data using the hue, size, and style parameters. These parameters control what visual semantics are used to identify the different subsets. It is possible to show up to three dimensions independently by using all three semantic types, but this style of plot can be hard to interpret and is often ineffective. Using redundant semantics (i.e. both hue and style for the same variable) can be helpful for making graphics more accessible."""

# LinePlot
sns.set_theme(style="white")
sns.lineplot(data=df_filter, x="Year Founded", y= "Growth Rate (%)")
plt.suptitle("Growth Rate by Year Founded")
plt.show()

"""Show point estimates and errors as rectangular bars.

A bar plot represents an aggregate or statistical estimate for a numeric variable with the height of each rectangle and indicates the uncertainty around that estimate using an error bar. Bar plots include 0 in the axis range, and they are a good choice when 0 is a meaningful value for the variable to take."""

# BarPlot
sns.set_theme(style="whitegrid")
sns.barplot(data=df_filter, x="Startup Name", y="Growth Rate (%)", legend=False)
plt.suptitle("Growth Rate by Startup")
plt.show()

"""Figure-level interface for drawing categorical plots onto a FacetGrid.

This function provides access to several axes-level functions that show the relationship between a numerical and one or more categorical variables using one of several visual representations. The kind parameter selects the underlying axes-level function to use."""

# CatPlot
sns.set_theme(style="whitegrid")
sns.catplot(data=df_filter, x="Startup Name", y="Growth Rate (%)", kind="box")
plt.suptitle("Growth Rate by Startup")
plt.show()

"""Plot rectangular data as a color-encoded matrix.

This is an Axes-level function and will draw the heatmap into the currently-active Axes if none is provided to the ax argument. Part of this Axes space will be taken and used to plot a colormap, unless cbar is False or a separate Axes is provided to cbar_ax."""

# HeatMap
sns.set_theme(style="darkgrid")
glue = df_filter.pivot(columns="Investment Amount (USD)", values="Valuation (USD)")
sns.heatmap(glue)
plt.show()

read = input("Wait for me....")














