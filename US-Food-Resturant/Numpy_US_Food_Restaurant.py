import numpy as np


# Load FastFoodRestaurants (4).csv dataset
address, city, country, lat, long = np.genfromtxt("FastFoodRestaurants (4).csv", delimiter=",", usecols=(0, 1, 2, 4, 5), unpack=True, dtype=str, skip_header=1 , )
print(f"Address is {address}")
print(f"City is {city}")
print(f"Country is {country}")
print(f"Latitude is {lat}")
print(f"Longitude is {long}")


# Clean latitude Remove Bad Rows
# Use try OR except 
latitude = []

for val in lat:
    try:
        latitude.append(float(val))
    except ValueError:
        continue



# US_Food_Restaurant (latitude) statistics operations
print("US_Food_Restaurant (latitude) mean" , np.mean(latitude))
print("US_Food_Restaurant (latitude) avarage" , np.average(latitude))
print("US_Food_Restaurant (latitude) std" , np.std(latitude))
print("US_Food_Restaurant (latitude) median" , np.median(latitude))

# Percentile
print("US_Food_Restaurant (latitude) percentile - 25" , np.percentile(latitude, 25))
print("US_Food_Restaurant (latitude) percentile - 75" , np.percentile(latitude, 75 ))
print("US_Food_Restaurant (latitude) percentile - 3" , np.percentile(latitude, 3))

# Minimum and Maximum Latitude
print("US_Food_Restaurant (latitude) max" , np.max(latitude))
print("US_Food_Restaurant (latitude) min" , np.min(latitude))

# Math Operations 
print("US_Food_Restaurant (latitude) square" , np.square(latitude))
print("US_Food_Restaurant (latitude) sqrt" , np.sqrt(latitude))
print("US_Food_Restaurant (latitude) power" , np.power(latitude, latitude))
print("US_Food_Restaurant (latitude) abs" , np.abs(latitude))


# Clean longitude Remove Bad Rows
longitude = []

for valu in long:
     try:
        longitude.append(float(valu))
     except ValueError:
         continue
            

# Find the smaller length between latitude and longitude arrays
min_length = min(len(latitude), len(longitude))

# Cut latitude and longitude array to match the minimum length
latitude = latitude[:min_length]
longitude = longitude[:min_length]

# Convert longitude and latitude in array
latitude = np.array(latitude)
longitude = np.array(longitude)



# Arithmetic Operation 
addition = longitude+latitude
subtraction = longitude-latitude
multiplication = longitude * latitude
division = longitude / latitude

print("Longitude and Latitude Addition ", addition)
print("Longitude and Latitude Subtraction ", subtraction)
print("Longitude and Latitude Multiplication ", multiplication)
print("Longitude and Latitude Division ", division)

# Trignometric Function
latpie = (latitude/np.pi) +1

# Now Calculate sin, cosine, and tengent
sine_values = np.sin(latpie)
cosine_values = np.cos(latpie)
tangent_values = np.tan(latpie)

print(f"Sin Values is {sine_values}")
print(f"Cosine Values is {cosine_values}")
print(f"Tangent Values is {tangent_values}")

# Calculate Natural Logrithm And Base 10 Logrithm
log_array = np.log(latpie)
log10_array = np.log10(latpie)

print(f"Natural Logrithm Values is {log_array}")
print(f"Base-10 Logrithm Values is {log10_array}")

# 2 Dimenstional Array
dime_2_array = np.array([latitude,longitude])
print("2 Dimenstional Array is ",dime_2_array)

# Check The Dimenstional Array 
print("Dimenstional array is ", dime_2_array.ndim)

# Chech Size Of Array 
print("Size Of 2D Array is ", dime_2_array.size)

# Check Shape Of Array
print("Shape Of Array Is ", dime_2_array.shape)

# Check The Datatype Of Array 
print("Data Type Of Array Is ", dime_2_array.dtype)

# Slicing Array
D2_lat_long_slice = dime_2_array[:1, :5]
print("Slicing Array is : ", D2_lat_long_slice)

# Syntax Of Slicing WIth Step [Start_Row : End_row, Start_col : End_col : Step]
D2_lat_long_slice1 = dime_2_array[:1, 4:15 :4]
print("Slicing Of Array is :" , D2_lat_long_slice1)

# Indexing Array
D2_lat_long_itemOnly = dime_2_array[0 , 1]
print("Indexing Of Array Is :", D2_lat_long_itemOnly)
D2_lat_long_itemOnly1 = dime_2_array[0 , 2]
print("Indexing Of Array Is :", D2_lat_long_itemOnly1)

# Loop through each individual element of the NumPy array D2LongLat
# np.nditer() allows us to iterate over multi-dimensional arrays as flat (1D) elements
for elem in np.nditer(dime_2_array):
    print("Element Of 2D Array Is :", elem)

# Iterate over each element in the D2LongLat array using np.ndenumerate
# np.ndenumerate returns both the index (as a tuple) and the element value
for index, elem in np.ndenumerate(dime_2_array):
    print(index,elem)

# Reshape (1,298)
D2LongLat1TO298 = np.reshape(dime_2_array, (1, 298))
print(D2LongLat1TO298)
print("Size is ", D2LongLat1TO298.size)
print("Size is ", D2LongLat1TO298.ndim)
print("Size is ", D2LongLat1TO298.shape)

print("The End")







