import numpy as np

# Load CSV File 
Startup_Name, Industry, Funding_Rounds,Investment_Amount, Valuation, Growth_Rate  = np.genfromtxt("startup_growth_investment_data (2).csv", delimiter=",", usecols=(0, 1, 2, 3, 4, 8), unpack=True, dtype=None, skip_header=1)
print(Startup_Name)
print(Industry)
print(Investment_Amount)
print(Funding_Rounds)
print(Valuation)
print(Growth_Rate)

# Statistic Operation (Investment Amount (USD))
# Calculate And Display Mean
print("Investment Amount Means : ")
print(np.mean(Investment_Amount))
print()

# Calculate And Display average
print("Investment Amount Avarage : ")
print(np.average(Investment_Amount))
print()

# Calculate And Display Standard Deviation
print("Investment Amount Std : ")
print(np.std(Investment_Amount))
print()

# Calculate And Display Median
print("Investment Amount Median : ")
print(np.median(Investment_Amount))
print()

# Calculate And Display Percentile (25)
print("Investment Amount Percentile - 25 : ")
print(np.percentile(Investment_Amount, 25))
print()

# Calculate And Display Percentile (75)
print("Investment Amount Percentile - 75 : ")
print(np.percentile(Investment_Amount, 75))
print()

# Calculate And Display Percentile (3)
print("Investment Amount Percentile - 3 : ")
print(np.percentile(Investment_Amount, 3))
print()

# Calculate And Display Maximum
print("Max Investment Amount : ")
print(np.max(Investment_Amount))
print()

# Calculate And Display Minimum
print("Min Investment Amount : ")
print(np.min(Investment_Amount))
print()

# Math Operations 
# Calculate Square
print("Investment Amount Square : ")
print(np.square(Investment_Amount))
print()

# Calculate Square Root
print("Investment Amount Square Root : ")
print(np.sqrt(Investment_Amount))
print()

# Calculate Power 
print("Investment Amount Power : ")
print(np.power(Investment_Amount,Investment_Amount))
print()

# Calculate Absolute Value 
print("Investment Amount  Absolute Value : ")
print(np.abs(Investment_Amount))
print()

# Basic Arithmetic Operation 
# Addition (Investment_Amount + Valuation)
addition = Investment_Amount + Valuation
print("Investment_Amount + Valuation : ")
print(addition)

# Subtraction (Investment_Amount - Valuation)
subtraction = Investment_Amount - Valuation
print("Investment_Amount - Valuation : ")
print(subtraction)

# Multipication (Growth_Rate * 2)
multiplication = Growth_Rate * 2
print("Growth_Rate * 2 : ")
print(multiplication)

# Division (Investment_Amount / Funding_Rounds)
division = Investment_Amount / Funding_Rounds
print("Investment_Amount / Funding_Rounds : ")
print(division)

# Trigonometric Functions
adjusted_investment = (Investment_Amount/np.pi) + 1

# Calculate Sine, Cosine And Tangent
# Calculate The Sine Of the Adjusted Investment Value
calc_sine = np.sin(adjusted_investment)
print("Sine Of Adjusted Investment Amount : ")
print(calc_sine)

# Calculate The Cos Of the Adjusted Investment Value
calc_cos = np.cos(adjusted_investment)
print("Cos Of Adjusted Investment Amount : ")
print(calc_cos)

# Calculate The Tangent Of the Adjusted Investment Value
calc_tangent = np.cos(adjusted_investment)
print("Tangent Of Adjusted Investment Amount : ")
print(calc_tangent)

# Calculate The Exponential Function Value Of the Adjusted Investment Value
calc_exp = np.exp(adjusted_investment)
print("Exponential Function Adjusted Investment Amount : ")
print(calc_exp)

# Calculate The Natural Logarithm And Base-10 Logarithm
# Calculate Natural Logarithm (Ln) Of Adjusted Investment Values
log_array = np.log(adjusted_investment)
print("Natural Logarithm (Ln) Adjusted Investment Values : ")
print(log_array)

# Calculate Base-10 Logarithm Of Adjusted Investment Values
log10_array = np.log10(adjusted_investment)
print("Base-10 Logarithm Adjusted Investment Values : ")
print(log10_array)

# 2 Dimentional Arrary
dimension_2_array = np.array([Investment_Amount, Growth_Rate])
print("2 Dimentional Arrary : ")
print(dimension_2_array)

# Check The Dimension Of Array 
print("Dimension Of Array : ")
print(dimension_2_array.ndim)

# Check The Size Of Array 
print("Size Of Array is : ")
print(dimension_2_array.size)

# Check The Shape Of Array 
print("Shape Of Array Is : ")
print(dimension_2_array.shape)

# Check The DataType Of Array
print("DataType Of Array is : ")
print(dimension_2_array.dtype)

# Splicing Array
# 2 Dimentional Arrary - Splicing Array - D2LongLat[:1,:5]
dimension_2_array_slice = dimension_2_array[:1, :5]
print("Splicing Array is : ")
print(dimension_2_array_slice)

# 2 Dimentional Arrary - Splicing Array - D2LongLat[:1, 4:15:4]
dimension_2_array_slice1 = dimension_2_array[:1, 5:20:5]
print("Splicing Array is : ")
print(dimension_2_array_slice1)

# Indexing Array
dimension_2_array_index = dimension_2_array[1,2]
print("2 dimentional arrary - Index Array - D2LongLatSlice[1,5] ")
print(dimension_2_array_index)

# Builtin Function nditer
for val in np.nditer(dimension_2_array):
    print(val)


for index, val in np.ndenumerate(dimension_2_array):
    print(index, val)



print("-End-")
























