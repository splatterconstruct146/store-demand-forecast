# Store Demand Forecast
A project that uses the Prophet ML model to predict future sales of stores using past sales data.\

## Motivation
This project was created as a response to difficulties in making predictions of time series data in another one of my projects. I found out about the Prophet model and wanted to do a test run here before trying it out in that previous project.

## Shortcomings
The Prophet model works great for this dataset - which doesn't have a large number of rows. This dataset has one row per day for 5 days (so this is only 1825 rows).
I'll be curious to test this out on a time-series data that has info by the minute for 3 years (so approx 1.5mil rows).

## Summary of Dataset
The dataset can be found here: [https://www.kaggle.com/competitions/demand-forecasting-kernels-only/data](https://www.kaggle.com/competitions/demand-forecasting-kernels-only/data)

A breakdown of the dataset columns is as follows:
* date: date of the sale data. No holiday effects or store closures.
* store: store ID
* item: item ID
* sales: Number of items sold at the particular store on a particular date

## Files
The files found in this repository and their descriptions are as follows:
1. train.csv.zip: Dataset containing all the info used for this project. 
2. StoreDemandForecast.ipynb: The jupyter notebook holding all the code and various descriptions/thought processes in testing the model.
3. Files inside the output folder:
   1. accuracy_of_model_by_store.png: A bar plot showing the accuracy rate (1-MAPE) of all 10 stores.
   2. mean_sales_all_stores_day_split.png: A scatterplot showing the average number of sales of all stores for each day of the year, for a year. Helps visualize number of sales by day and if there are any trends.
   3. mean_sales_all_stores.png: A lineplot showing the mean sales of all stores across the entire dataset period.
   4. mean_sales_forecast_components.png: An image with 3 subplots showing the overall trend over time, weekly trend, and yearly trend of the dataset. This image is created with Prophet.
   5. mean_sales_forecast.png: A line plot showing the mean sales across time of all stores, and a prediction for the sales in the desired future period. The black dots on the plot indicate the points used to train the model. This plot is created with Prophet.
   6. results.csv: A csv holding the output table of each store and the various measures of error associated to the model trained for each store against the test dataset. More on this below.

## Sample Screenshots
![mean_sales_all_stores](https://github.com/splatterconstruct146/store-demand-forecast/assets/135209633/e49a0d09-59bd-4185-a5fd-e4a871368ba0)

Figure 1: Mean sales of all stores across the entire period in the dataset. 2013-2017.

![mean_sales_forecast](https://github.com/splatterconstruct146/store-demand-forecast/assets/135209633/db452378-bb3e-4423-bdcb-40e68954295b)

Figure 2: Mean sales of all stores across 2013-2016. 2017 is the predicted mean sales of all stores using the 2013-2016 data as the training set.

## Process
Shows the process of creating and training the model for this project.

### 1. Cleaning
Luckily there were no NaN values so only needed to convert the date column from objects dtype to datetime64 dtype.

![Screenshot_20240106_210322](https://github.com/splatterconstruct146/store-demand-forecast/assets/135209633/d2feed5d-6afb-4029-9a67-04e984e8957c)

Image 1: Shows the code checking for null values in all columns. Also converting date column into datetime64 dtype.

There were also no duplicate rows - so the dataset is good to go.

![Screenshot_20240106_210659](https://github.com/splatterconstruct146/store-demand-forecast/assets/135209633/29afa2c0-c743-48f6-aab8-d8d94b90e8c6)

Image 2: Shows the code checking for duplicated rows among date, store, item columns. The following code checks for overall duplicate rows (carbon copies of another row)

### 2. Data Exploration
Found out that there were 10 unique stores and 50 unique items for each store in the dataset. Note there were 1826 unique dates - this implies about 5 years worth of data.

![Screenshot_20240106_210930](https://github.com/splatterconstruct146/store-demand-forecast/assets/135209633/16fda4e6-31d0-4c68-a103-ba1cc771b561)

Image 3: Code showing the number of unique dates, stores, items.

Checked out the mean number of sales of all stores for each day. This also served to visually identify any trends or patterns.

![Screenshot_20240106_211349](https://github.com/splatterconstruct146/store-demand-forecast/assets/135209633/e79b0585-9355-4f62-9d6c-69f21f1824a5)

Image 4: Code for finding the mean number of sales of all stores. Code below shows the code to create the plot shown in Figure 1.

The code below shows the code for creating the scatterplot that shows the mean number of sales by day for the 2015 year.

![Screenshot_20240106_211709](https://github.com/splatterconstruct146/store-demand-forecast/assets/135209633/a15301f7-a3ac-43cb-a15e-e7d79fdcbe09)

Image 5: Code for creating the scatterplot in Figure 3.

![mean_sales_all_stores_day_split](https://github.com/splatterconstruct146/store-demand-forecast/assets/135209633/90756165-444c-45fa-92e1-b76984417944)

Figure 3: Scatterplot showing mean number of sales of all stores, broken down by days in 2015. We can see highest number of sales on Sun, and lowest on Mon.

### 3. Creating the Model
![Screenshot_20240106_212032](https://github.com/splatterconstruct146/store-demand-forecast/assets/135209633/277a1b60-058e-4c6d-8c11-8e2daf2d68da)

Image 6: Creating a copy dataframe so the original remains untouched.

![Screenshot_20240106_212150](https://github.com/splatterconstruct146/store-demand-forecast/assets/135209633/cdebb608-29e4-4375-a652-b684f2ed5bc4)

Image 7: Splitting up the dataset into train (which was used to train the model) and test (which was used to compare the trained model to check how similar or different the predicted model is from the actual).

### 4. Error Checking
The final step is to check how accurate the trained model is.

![Screenshot_20240106_212503](https://github.com/splatterconstruct146/store-demand-forecast/assets/135209633/f04ef1b7-acd1-4b79-bf3f-47b33b42c427)

Image 8: The code snippet above shows the 4 measures of error used to evaluate this model again the test dataset made earlier. Overall the model performed quite well.

### 5. Mass Testing on all Stores
Now as a form of sanity check, tested the model on all stores to see if there's any wildly different results.

![Screenshot_20240106_212834](https://github.com/splatterconstruct146/store-demand-forecast/assets/135209633/6c8f6e3c-3095-4b63-bf7b-7ebda27655c4)

Image 9: The code above shows a for loop to create a model for each store in the dataset, and to test the measure of error of each store's model against the test set.

Created a new column that helped visual clarity. It's just 1-MAPE to get the accuracy rate.

![Screenshot_20240106_213253](https://github.com/splatterconstruct146/store-demand-forecast/assets/135209633/87a39d5a-cc80-45c7-bdca-996ce0412217)

Image 10: Code to create a new column in the dataframe.

Some statistics of the results dataframe.

![Screenshot_20240106_213443](https://github.com/splatterconstruct146/store-demand-forecast/assets/135209633/38b265b4-021f-4519-8947-dfdaf37e41e9)

Image 11: Table of statistics of the results dataframe.

## Conclusion
In conclusion Prophet works well for this particular dataset. Since results were promising this time around, will try it out in the other project to see if it works for that usecase. 

