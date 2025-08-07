import pandas as pd

# Read CSV File To DataFrame
df = pd.read_csv("FastFoodRestaurants (4).csv", delimiter=",")
print("Data_Frame is :", df)
print(f"Data Type Of - DF is :{df.dtypes}")
print(f"Info() Is : {df.info()}")

# Print Last Five Rows
print("Last Throw Of DataFrame")
print(df.tail(5))
print()

# Print First Five Rows 
print("First Five Rows Of DataFrame")
print(df.head(5))
print()

# Describe() Method 
print("Summary of Statistics of DataFrame using describe() method", df.describe())

# Counting The Rows And Columns In DataFrame Using Shape() Method
print("Counting The Rows And Columns In DataFrame Using Shape() Method ", df.shape)
print()

# Access The City Column From Df
single_col = df["city"]
print("Access The City Column From Df", single_col)
print()

# Access The Multiple Column From Df
multiple_col = df[["city", "country"]]
print("Access The Multiple Column From Df ", multiple_col)
print()

# Selecting A Single Row Using .loc
single_row = df.loc[1]
print("Selecting A Single Row Using .loc", single_row)
print()

# Selecting A Multiple Row Using .loc
multiple_row = df.loc[[1,3]]
print("Selecting A Multiple Row Using .loc", multiple_row)

# Selecting A Slicing Of Rows Using .loc
slicing_row = df.loc[1:5]
print("Selecting A Slicing Of Rows Using .loc", slicing_row)
print()

# Conditional Selection Of Rows Using .loc
conditional_select = df.loc[df["city"] == "Athens"]
print("Conditional Selection Of Row ", conditional_select)
print()

# Selecting A Single Column Using .loc
single_col_1 = df.loc[:,"city"]
print("Selecting A Single Column Using .loc", single_col_1)
print()

# Selecting A Multiple Column Using .loc
multiple_col_1 = df.loc[:, ["city", "country"]]
print("Selecting A Multiple Column Using .loc ", multiple_col_1)

# Selecting A Slicing Of Column Using .loc 
slicing_col = df.loc[:,"address":"city"]
print("Selecting A Slicing Of Column Using .loc ", slicing_col)
print()

# Combined Row And Column Selection Using .loc
combined_row_col = df.loc[df["city"] == "Athens", "city" : "name"]
print("")


# Now Case 2 .iloc

# Selectig A Single Row Using .iloc
single_row_1 = df.iloc[0]
print("Selectig A Single Row Using .iloc", single_row_1)
print()

# Selectig A Multiple Row Using .iloc
multiple_row_1 = df.iloc[[0,2,3]]
print("Selectig A Multiple Row Using .iloc", multiple_row_1)
print()

# Selecting A Single Column Using .iloc 
single_col_2 = df.iloc[:,2]
print("Selecting A Single Column Using iloc ", single_col_2)
print()

# Selecting A Multiple Column Using .iloc
multiple_col_2 = df.iloc[:, [2,5]]
print("Selecting A Multiple Column Using .iloc")
print()

# Slicing Of Column Using .iloc
slicing_col_1 = df.iloc[:, 2:5]
print("Slicing Of Column Using .iloc", slicing_col_1)
print()

# Combined Rows And Column using .iloc
combined_row_col_1 = df.iloc[[1,2,4], 2:5]
print("Combined Rows And Column using .iloc ", combined_row_col_1)
print()

# Now Case 3
# Add A New Row To A Pandas DataFrame
# Chech The Len Of Column
print(len(df.columns))

df.loc[len(df.index)] = ["324 Main St California", "California", "US", "us/ny/massena/324mainst/-1161002137", 45.9213, -74.89021, "McDonald's", 13662, "California", "http://mcdonalds.com,http://www.mcdonalds.com/?cid=RF:YXT_FM:TP::Yext:Referral"]
print(df)

# Remove Rows And Column From A Pandas DataFrame

# Deleting Row With Index 1
df.drop(1, axis=0, inplace=True)

# Deleting Row With Index 4
df.drop(index=5, inplace=True)

# Deleting Row With Index 8 And 9
df.drop([8,9], axis=0, inplace=True)

print("Check Remove Rows ")
print(df)

# Now Remove Column 
# Delete PostalCode Column
df.drop("postalCode", axis=1, inplace=True)

# Delete Marital Status Column
df.drop(columns="country", inplace=True)

# Delete Multiple Column 
df.drop(["websites","province"], axis=1, inplace=True)

print("Check Delete Column ")
print(df)

# Rename Labels/Column In A DataFrame

# Rename Column 'Name' To 'First_Name'
df.rename(columns={"city" : "city_name"}, inplace=True)

# Rename Multiple Column 'Name' To 'First_Name' 
df.rename(columns={"keys":"complete_key", "latitude":"latitude_1"}, inplace=True)

print("Check Rename Column")
print(df)

# Rename Row Labels In A DataFrame

# Rename Row {0 : 1} 
df.rename(index={0:3}, inplace=True)

# Rename Multiple Row {10 : 15 , 15:200}
df.rename(index={10:15, 15:200}, inplace=True)

print("Check Rename Rows ")
print(df)

# Query() To Select Data
# Select The Rows Where The Latitude Is Greater Than 44.9213
selected_rows = df.query("city_name == \"Massena\" or latitude_1 > 44.9213")
print(selected_rows.to_string())
print(len(selected_rows))

# Sort DataFrame By Latitude In Ascending Order
sorted_df = df.sort_values(by="latitude_1")
print(sorted_df.to_string(index=False))


