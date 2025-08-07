import numpy as np

brokered_by , price , acre_lot , city , house_size = np.genfromtxt("RealEstate-USA.csv" , delimiter="," , usecols=(0 , 2 , 5 , 7 ,10) , unpack=True , dtype=None , skip_header=1)
print(brokered_by)
print(price)
print(acre_lot)
print(city)
print(house_size)


#statistic opperation 
#price
print("Real state price mean : " , np.mean(price))
print("Real state price avarage : " , np.average(price))
print("Real state price std : " , np.std(price))
print("Real state price median : " , np.median(price))
print("Real state price percentile 25 : " , np.percentile(price ,25))
print("Real state price percentile - 75 : " , np.percentile(price ,75))
print("Real state price percentile - 3 : " , np.percentile(price , 3))
print("Real state price min : " , np.min(price))
print("Real state price max : " , np.max(price))

#statistic opperation 
#house_size
print("Real state house_size mean : " , np.mean(house_size))
print("Real state house_size avarage : " , np.average(house_size))
print("Real state house_size std : " , np.std(house_size))
print("Real state house_size median : " , np.median(house_size))
print("Real state house_size percentile - 25 : " , np.percentile(house_size , 25))
print("Real state house_size percentile - 75 : " , np.percentile(house_size , 75))
print("Real state house_size percentile - 3 : " , np.percentile(house_size , 3))
print("Real state house_size min : " , np.min(house_size))
print("Real state house_size max : " , np.max(house_size))

# Addition and Substraction or Multiplication
addition = price + house_size
substraction = price - house_size
multiplication = price * house_size

print("Addition of price and house_size is : " ,addition )
print("Addition of price and house_size is : " ,substraction )
print("Addition of price and house_size is : " ,multiplication )

# 2D array
array = np.array([price , house_size])
print("2d array is : " , array)
print("dimenstiin is : " , array.ndim)

# 3D array
array1 = np.array([[house_size , price , acre_lot]])
print("3D array is : " , array1)
print("dimenstion is :" , array1.ndim)

# iterate the array of price
array2 = np.array(price)
print("1D array is :" , array2)
print("dimenstion is :" , array2.ndim)

for i in np.nditer(array2):
    print(f"Your price is : {i}" )

# itrate the array of price using np.ndenumerate ()   
for i in np.ndenumerate(array2):
    print(f"Price is : {i}")

# 7 common properties of array 
print("Dimenstion of array is :" , array2.ndim )
print("Size of array is : ", array2.size)
print("Shape of array is : " , array2.shape)
print("dtype of array is : " , array2.dtype)
print("itemsize of array is : " , array2.itemsize)
print("nbytes of array is : " , array2.nbytes)
print("transpose of array is : " , array2.T)

# slicing of 2D array 
slicing_array = array[1:3,2:4]
print("Slicing of array is : " , slicing_array)

slicing_array1 = array[2:8,3:5]
print("Slicing of 2D array is : " , slicing_array1)

# Geomatric operation in numpy
geo_cos = np.cos(array)
print("Cos is : " , geo_cos)

geo_sin = np.sin(array)
print("sin is : " , geo_sin)



