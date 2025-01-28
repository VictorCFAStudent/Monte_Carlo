import numpy as np
import pandas as pd
import yfinance as yf
from matplotlib import pyplot as plt

#This is a Monte Carlo simulation program for stock prices

#We are going to create a Stock class to properly structure the code

class Stock:

    def __init__(self, stock, start_date, end_date):  
        self.ticker = stock
        self.start_date = start_date
        self.end_date = end_date
        self.dataframe = yf.Ticker(self.ticker).history(start = self.start_date, end = self.end_date) #Getting 2024 stock data
        self.number_of_days = self.dataframe.shape[0] - 2 #-2 because first and last rows won't get Log Returns Value

    def preparing_df(self):

        # Checking if there are some Stock Splits and deleting the column to simplify the simulation
        if 'Stock Splits' in self.dataframe.columns:
            stock_splits_count = self.dataframe['Stock Splits'].sum() #Checking
            if stock_splits_count >= 1:
                print(f"Warning, {stock_splits_count} stock split(s) will be deleted to simplify the simulation.")
            self.dataframe.drop(columns=['Stock Splits'], inplace=True)#Deleting

        # Checking and replacing NaN values by column mean
        for col in self.dataframe.columns:
            if self.dataframe[col].isnull().sum() > 0:
                print(f"{col} contains NaN values, replacing with mean.")
                self.dataframe[col].fillna(self.dataframe[col].mean(), inplace=True)
        return self.dataframe
    
    def log_returns_calulation(self, col):
        self.dataframe['Log Returns'] = np.log(self.dataframe[col] / self.dataframe[col].shift(1))
        return self.dataframe['Log Returns']

def intializing_matrix(random_seed, matrix_shape):   
    np.random.seed(random_seed)
    matrix = np.random.normal( 0, 1, matrix_shape)
    initial_price = stock.dataframe['Close'].iloc[-1]
    matrix[0] = initial_price
    
    return matrix
 
def simulation(drift, volatility, matrix):
    exp_drift = np.exp(drift)
    for t in range(1, matrix.shape[0]):
        matrix[t] = matrix[t-1] * exp_drift * np.exp( volatility *matrix[t] )
    return paths_matrix

def confidence_interval(mean, std, confidence_level):
    if confidence_level == 99 :
        z = 2.567
    elif confidence_level == 95 :
        z = 1.96
    else :
        print("Input 99 or 95 confidence level")
        return None
    lower_bound = max(0, mean - z * std)  # Ensure lower bound is not negative
    upper_bound = mean + z * std
    return lower_bound, upper_bound

#Getting Stock data
ticker_name = input("Please input the ticker : ")
start_date = input("Please input the start date of the historical data period used for calculations : ")
end_date = input("Please input the end date of the historical data period used for calculations : ")

stock = Stock(ticker_name, start_date, end_date)
stock_df = stock.preparing_df()
log_returns = stock.log_returns_calulation('Close')
days = stock.number_of_days

print(stock_df)

#Getting Log Returns of the stock
mean_log_returns = np.mean(log_returns)
sum_squared_log_returns = np.sum(np.square(log_returns))

#Getting parameters needed for the simulation (drift, volatility and number of paths)
historical_volatility = np.sqrt( (1 / (days-1) ) * sum_squared_log_returns)
drift = ( mean_log_returns - (np.square(historical_volatility)/2) )
paths = 1000

#Initializing paths_matrix, with row = days and column = price path
#1st row contains Initial Price, the rest of the Matrix contains random values
paths_matrix = intializing_matrix(10, (stock_df['Log Returns'].count(), paths))

#Simulating the paths
paths_matrix = simulation(drift, historical_volatility, paths_matrix)

#Getting Column which contains final price for each path and calculating mean and std
final_prices = paths_matrix[-1]
mean_final_prices = np.mean(final_prices)
std_final_price = np.std(final_prices)

#We get Column mean of the paths so we can plot the mean path
mean_paths = np.mean(paths_matrix, axis=1)

#Visualizing the simulations
plt.figure(figsize=(14,10))
plt.plot(paths_matrix, alpha = 0.5)
plt.title(f"Monte Carlo simulation for {stock.ticker} stock price in 2025 ({paths} paths)", fontsize = 12, weight = 'bold')
plt.xlabel("Time (Days)", weight = 'bold')
plt.ylabel("Stock Price", weight = 'bold')
plt.annotate(f"Mean Final Price : {mean_final_prices}", (-5, np.max(final_prices)*(97/100)) ) #Getting the estimated price on the figure
plt.plot(mean_paths, color="black", linewidth = 2, label="Mean Path")
plt.grid(True, alpha = 0.3)
plt.legend()
plt.show()

#Confidence Interval (95 and 99)
CI_95 = confidence_interval(mean_final_prices, std_final_price, 95)
CI_99 = confidence_interval(mean_final_prices, std_final_price, 99)

print(f"Estimated price: {round(mean_final_prices,4)}")
print(f"CI 95% for final price : [{max(0, CI_95[0])}, {CI_95[1]}]")
print(f"CI 99% for final price :[{max(0, CI_99[0])}, {CI_99[1]}]")
