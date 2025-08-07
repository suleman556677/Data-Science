import pandas as pd

# Load CSV File 
df = pd.read_csv("startup_growth_investment_data (2).csv", delimiter=",")

# Display DataFrame
print("Data - Fram : ")
print(df)

# Check The DataType Of DataFrame
print("DataType Of DataFrame : ")
print(df.dtypes)

# Check The Shape Of DataFrame
print("Shape Of DataFrame : ")
print(df.shape)

# Display info()
print("Info Of DataFrame : ")
print(df.info())

# Display Last Five Rows Of DataFrame
print("Last Five Rows Of DataFrame : ")
print(df.tail(5))

# Display First Five Rows Of DataFrame
print("First Five Rows Of DataFrame : ")
print(df.head(5))

# Summary Of Statistics Of Data Frame Using Describe() Method
print("Summary Of Statistics Of Data Frame Using Describe() Method :")
print(df.describe())

# Access The Name Of Column
Industry = df["Industry"]
print("Access The Name Of Column : ")
print(Industry)
print()

# Access The Multiple Name Of Column
Industry_1 = df[["Industry", "Country"]]
print("Access The Multiple Name Of Column : ")
print(Industry_1)
print()

# Case 1 
# Selecting A Single Row Using .loc
single_row = df.loc[0]
print("Selecting A Single Row Using .loc : ")
print(single_row)
print()

# Selecting A Multiple Rows Using .loc
multiple_row = df.loc[[2,4]]
print("Selecting A Multiple Rows Using .loc : ")
print(multiple_row)
print()

# Selecting A Slice Of Rows Using .loc
slicing_row = df.loc[1:9]
print("Selecting A Slice Of Rows Using .loc : ")
print(slicing_row)
print()

# Conditional Selection Of Rows Using .loc
con_select = df.loc[df["Industry"] == "EdTech"]
print("Conditional Selection Of Rows Using .loc : ")
print(con_select)
print()

# Selecting A Single Column Using .loc
single_col = df.loc[:, "Country"]
print("Selecting A Single Column Using .loc : ")
print(single_col)
print()

# Selecting A Mutiple Columns using .loc
multiple_col = df.loc[:, ["Industry", "Country"]]
print("Selecting A Mutiple Column using .loc : ")
print(multiple_col)
print()

# Selecting A Slice Of Columns Using .loc
slicing_col = df.loc[:1, "Industry":"Country"]
print("Selecting A Slice Of Columns Using .loc : ")
print(slicing_col)
print()

# Combined Row And Column Selection Using .loc
com_row_col = df.loc[df["Industry"] == "EdTech", "Industry":"Country"]
print("Combined Row And Column Selection Using .loc : ")
print(com_row_col)
print()

# Case 2 : Using .iloc - Starts Here
# Selecting A Single Row Using .iloc
single_row_1 = df.iloc[0]
print("Selecting A Single Row Using .iloc : ")
print(single_row_1)
print()

# Selecting A Multiple Rows Using .iloc
multiple_row_1 = df.iloc[[0,5,7]]
print("Selecting A Multiple Row Using .iloc : ")
print(multiple_row_1)
print()

# Selecting A Slice Rows Using .iloc
slicing_row_1 = df.iloc[1:10]
print("Selecting A Slice Rows Using .iloc : ")
print(slicing_row_1)
print()

# Selecting A Single Column Using .iloc
single_col_1 = df.iloc[:1, 2]
print("Selecting A Single Column Using .iloc : ")
print(single_col_1)
print()

# Selecting A Multiple Column Using .iloc
multiple_col_1 = df.iloc[:1, [2,5]]
print("Selecting A Multiple Column Using .iloc : ")
print(multiple_col_1)
print()

# Selecting A Slice Of Column Using .iloc
slicing_col_1 = df.iloc[:1, 2:4]
print("Selecting A Slice Of Column Using .iloc : ")
print(slicing_col_1)
print()

# Combined Row And Column Selection Using .iloc
com_row_col_1 = df.iloc[[1,3,7], 3:6]
print("Combined Row And Column Selection Using .iloc : ")
print(com_row_col_1)
print()

# Next Run
print("Next Run")
print()

""""Pandas DataFrame Manipulation
DataFrame manipulation in Pandas involves editing and modifying existing DataFrames. Some common DataFrame manipulation operations are:

Adding rows/columns
Removing rows/columns
Renaming rows/columns"""

# Add A New Row To A Pandas DataFrame
df.loc[len(df.index)] = ["Startup_5001", "Biotech", 5, 1208766602.64, 2265635198.775511, 8, "USA", 2010, 71.09 ]
print("Modified DataFrame : ")
print(df)
print()

# Deleting Row With Index 1
df.drop(1, axis=0, inplace=True)
print("Modified DataFrame - Remove Rows:")
print(df)
print()

# Deleting Row With Index 1
df.drop(index=2 , inplace=True)
print("Modified DataFrame - Remove Rows:")
print(df)
print()

# Deleting Rows With Index 6 And 8
df.drop([6, 8], axis=0, inplace=True)
print("Modified DataFrame - Remove Rows:")
print(df)
print()

# Delete (Number of Investors) Column
df.drop("Number of Investors", axis=1, inplace=True)
print("Modified DataFrame - Delete Column:")
print(df)
print()

# Delete Marital Status Column
df.drop(columns="Year Founded" , inplace=True)
print("Modified DataFrame - Delete Column:")
print(df)
print()

# Delete Multiple Column 
df.drop(["Startup Name", "Country"], axis=1, inplace=True)
print("Modified DataFrame - Delete Column:")
print(df)
print()

# Rename Labels In A DataFrame
# Rename Column "Name" To "First_Name"
df.rename(columns={"Industry":"Industry_Name"}, inplace=True)
print("Modified DataFrame  - Rename Labels :")
print(df)
print()

# Rename Row Labels
df.rename(index={0:8}, inplace=True)
print("Modified DataFrame - Rename Row - 0  >>> 7 , 1 >>> 10 , 2 >>> 100  Labels:")
print(df)
print()

# Rename Columns Multiple Index Labels
df.rename(mapper={5:6, 3:9}, axis=0, inplace=True)
print("Modified DataFrame - Rename Row - 0  >>> 7 , 1 >>> 10 , 2 >>> 100  Labels:")
print(df)
print()

# Query() To Select Data
# Select The Rows Where The Age Is Greater Than 25

""" select_querry = df.query("Industry_Name == \"Blockchain\" or Growth Rate (%) > 104.98")
print(select_querry.to_string())
print(len(select_querry))"""

# Sort DataFrame By Price In Ascending Order
sorted_df = df.sort_values(by="Growth Rate (%)")
print(sorted_df.to_string(index=False))

# Group The DataFrame By The Location_id Column And
# Calculate The Sum Of Price For Each Category
grouped = df.groupby("Funding Rounds")["Investment Amount (USD)"].sum()
print(grouped.to_string())
print("grouped :" , len(grouped))

# Use Dropna() To Remove Rows With Any Missing Values
df_cleaned = df.dropna()
print("Cleaned Data:\n",df_cleaned)

# Filling NaN Values With 0
df.fillna(0, inplace=True)
print("\nData After Filling NaN Values With 0:\n", df)






