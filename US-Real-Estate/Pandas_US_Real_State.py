import pandas as pd

# Load CSV file
df = pd.read_csv("RealEstate-USA.csv", delimiter="," )
print(df)

# Method/properties of datafram
print(df.info())
print(df.dtypes)
print(df.describe())
print(df.shape)

# DataFrame .to_string() method 
print(df.to_string(buf=None))
print(df.to_string(columns=["price"] , col_space=15 , header= ["price"] ))
print(df.to_string(na_rep="---"))
print(df.to_string(formatters={'price': '${:,.2f}'.format}))
print(df.to_string(justify="left"))
print(df.to_string(max_rows=5 , max_cols=2))

# Select top rows 
print(df.head(7))

# Select bottom rows
print(df.tail(9))

# Access the single column 
city = df["city"]
print("access the name of column : " , city)

street = df["street"]
print("access the name of column : " , street)

# Access the multiple column
city1 = df[["city" , "street"]]
print("both coloum are :" , city1)

# Selecting a single row using .loc
single_row = df.loc[5]
print("Single row : " , single_row)

# Selecting a multiple row using .loc
multiple_row = df.loc[[3 , 5 , 7]]
print("Multiple row using loc : " , multiple_row)

# selecting a slicing of row using .loc
slicing = df.loc[3:9]
print("slicing of row is : " , slicing)

# Conditional selection of rows using .loc
con_selction = df.loc[df["price"] > 100000]
print("Price is : " , con_selction)

# Conditinal selection of rows using .loc
con_selction1 = df.loc[df['city'] == "Adjuntas"]
print("Conditional selection is : " , con_selction1)

# Conditinal selection of rows using .loc
con_selction2 = df.loc[(df['city'] == "Adjuntas" ) & (df["price"] < 180500)]
print("Conditional selection is : " , con_selction2)

# Selecting a single row using .loc 
single_column = df.loc[7, ["city", "price", "street", "zip_code", "acre_lot"]]
print("Single column is : " , single_column)

# Selecting a slice of columns using .loc
slicing1 = df.loc[0:1 , "city":"zip_code"]
print("Slicing is : ", slicing1)

#Combined row and column selection using .loc
com_row_col = df.loc[df["city"] == "Adjuntas" ,"city" : "zip_code"]
print("combined row and column : " , com_row_col)

# Selecting a single row using .iloc
# Case 2
single_row1 = df.iloc[5]
print("Single row using .iloc " , single_row1)

# Selecting multiple rows using .iloc
multiple_row1 = df.iloc[[7 , 9 ,15]]
print("Multiple row is " , multiple_row1)

# Selecting a slice of  rows using .iloc
slicing2 = df.iloc[5:13]
print("Slicing of row using iloc" , slicing2)

# Selecting a single column using .iloc
single_column1 = df.iloc[:,3]
print("Single column using i loc : " , single_column1)

# Selecting multiple columns using .iloc
multiple_col = df.iloc[:,[2 ,4 , 7]]
print("Multiple column using iloc" , multiple_col)

# Selecting a slice of columns using .iloc
slicing3 = df.iloc[:,2:5]
print("Slicing column " , slicing3)

# Combined row and column selection using .
com_row_col1 = df.iloc[2:6,2:4]
print("Combined row and column using iloc " , com_row_col1)

# Combined row and column selection using .
com_row_col1 = df.iloc[2:6,2:4]
print("Combined row and column using iloc " , com_row_col1)

# print("Total columns:", len(df.columns))
# print(df.columns)

# Add a New Row to a Pandas DataFrame 
df.loc[len(df.index)] = [312392, "forsale" ,200000 , 3 ,2 , 0.12, 1962662 , "Adjuntas" , "Puerto Rico" , 602 , 920 ,1223 ]
print(df)

# delete row with index 2 
df.drop(2 , axis=0 , inplace=True)
print("deleting row is " , df)
print()

# delete row with index from 4 to 7th row 
df.drop([4,7] , axis=0 , inplace=True)
print(df)
print()

# Delete “house_size” column
df.drop("house_size" , axis=1 , inplace=True)
print(df)

# Delete “house_size” and “state” columns 
df.drop(["house_size", "state"], axis=1, inplace=True , errors='ignore')
# display the modified DataFrame after deleting rows
print("Modified DataFrame -  delete page_url ,property_type , location , city , column :")
print(df)


# Rename column “state”  to “state_Changed
df.rename(columns= {"state": "state_changed" ,"price":"price_changed" ,"bed":"bed_changed"}, inplace=True)
print(df)

# Rename label from 3 to 5
df.rename(mapper={3:5} , axis=1, inplace=True)
print(df)

# Querry()
if 'price' in df.columns and 'city' in df.columns:
    selected_rows = df.query('city != "adjuntas" or price > 11000000')
    print(selected_rows)
else:
    print(" 'price' or 'city' column not found in DataFrame!")

# Sort Value
df = pd.read_csv("RealEstate-USA.csv", delimiter="," )
print(df)

sorted_rows = df.sort_values(by="price")
print(sorted_rows)


# groupby
grouped = df.groupby('city')['price'].sum()

print(grouped.to_string())
print("grouped :" , len(grouped))

# Dropna()
df_cleaned = df.dropna()
print("Cleaned Data:\n",df_cleaned)

# filling NaN values with 0
df.fillna(0, inplace=True)