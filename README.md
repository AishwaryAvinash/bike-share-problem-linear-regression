# Demand Analysis for Shared Bikes Using Linear Regression

## Project Description
This project aims to predict the demand for shared bikes by analyzing historical data from BoomBikes, a US-based bike-sharing provider. With the business facing challenges due to revenue dips during the COVID-19 pandemic, the goal is to develop a model that helps understand the key factors affecting bike demand. The model can be used by management to make informed decisions and optimize business strategies post-pandemic. This project employs Python and machine learning techniques to create a multiple linear regression model.

## Table of Contents
- [Project Structure](#project-structure)
- [Installation Instructions](#installation-instructions)
- [Libraries Used](#libraries-used)
- [Data Description](#data-description)
- [Model Building](#model-building)
- [Evaluation](#evaluation)
- [Conclusion and Recommendations](#conclusion-and-recommendations)

## Project Structure
- `final copy ml.ipynb` - Jupyter notebook containing the code for data analysis, model building, and evaluation.
- `Assignment final.pdf` - Contains answers to questions asked in the problem statement.
- `README.md` - Project overview and documentation (this file).

## Installation Instructions
1. Clone the repository or download the project files.
2. Ensure you have Python 3.10 installed.
3. Open the Jupyter notebook `Bike_Demand_Prediction.ipynb` and run the cells in sequence.

## Libraries Used
The following libraries were used to perform the analysis and model building:
```python
- pandas 
- numpy 
- seaborn as sns 
- matplotlib.pyplot as plt
- warnings
- sklearn.model_selection
- sklearn.preprocessing
- sklearn.linear_model
- sklearn.metrics
- sklearn.feature_selection
- statsmodels.api
- statsmodels.stats.outliers_influence
```

## Data Description
The dataset used in this project consists of historical bike rental data from BoomBikes. Key features of the dataset include:

- **instant**: record index
- **dteday**: date
- **season**: season (1:spring, 2:summer, 3:fall, 4:winter)
- **yr**: year (0: 2018, 1:2019)
- **mnth**: month ( 1 to 12)
- **holiday**: weather day is a holiday or not (extracted from http://dchr.dc.gov/page/holiday-schedule)
- **weekday**: day of the week
- **workingday**: if day is neither weekend nor holiday is 1, otherwise is 0.
+ **weathersit**: 
  - *1: Clear, Few clouds, Partly cloudy, Partly cloudy*
  - *2: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist*
  - *3: Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds*
  - *4: Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog*
- **temp**: temperature in Celsius
- **atemp**: feeling temperature in Celsius
- **hum**: humidity
- **windspeed**: wind speed
- **casual**: count of casual users
- **registered**: count of registered users
- **cnt**: count of total rental bikes including both casual and registered

## Exploratory Data Analysis (EDA)
Before building the model, we performed Exploratory Data Analysis to gain insights into the dataset and understand the relationships between various features. Key steps included:

- **Data Visualization**: Used visualization tools such as Seaborn and Matplotlib to plot relationships between variables, such as the effect of temperature and humidity on bike rentals.
- **Correlation Analysis**: Evaluated the correlation between different features to identify significant predictors of bike demand.
- **Distribution Analysis**: Analyzed the distribution of the target variable (`count`) with respect to different attributes, to draw useful insights.

These insights guided the feature selection process and informed our modeling strategy.

## Model Building
The model building process involved several steps. Roughly summerizing them all:

1. **Data Preprocessing**:
   - Handling missing values and outliers in the dataset.
   - Scaling numerical features using MinMaxScaler for improved model performance.
   - Selecting significant features using Recursive Feature Elimination (RFE) to reduce model complexity and enhance interpretability.

2. **Data Splitting**:
   - The dataset was split into training and test sets using the `train_test_split` function to ensure a fair evaluation of model performance.

3. **Model Development**:
   - A multiple linear regression model was built using the `LinearRegression` class from the `sklearn` library.
   - The model was trained on the training set and validated on the test set to assess its performance.
   - Various iterations of the regression model were tested to identify the best-performing one.

## Evaluation
The model's performance was evaluated using the following metrics:

- **R-squared (R²)**: Indicates the proportion of variance in the dependent variable explained by the independent variables.
   - Training R²: 0.798
   - Test R²: 0.728
- **p-value**: Is kept less than 0.05 to prove that the independent variables have significant impact on bike demand.
- **VIF**: Is kept less than 2 to keep multicolinearity in check.

These metrics demonstrate the model's ability to generalize and predict bike demand effectively.

## Conclusion and Recommendations

- Year and September Boost Business: The analysis shows that the year (yr) and the month of September have a significant positive impact on bike rentals, leading to higher demand. It is recommended to increase bike availability and marketing efforts in September to capitalize on this trend.

- Weather Impact on Demand: Sales decrease significantly on cloudy and rainy days, as unfavorable weather conditions like rain make it harder for people to ride. Focusing on clear, non-rainy days to maximize bike availability could lead to higher usage.

- Windspeed’s Negative Effect: High windspeed negatively affects business, as it impacts riders' stability. It would be beneficial to adjust bike availability and offer incentives on days with lower windspeed.

- Lower Demand in Late Fall/Winter: Rentals drop significantly in colder months like November and December, likely due to uncomfortable riding conditions. Scaling back operations during these months could help optimize resources.

- Spring and Holidays See Fewer Rentals: There is a dip in sales during the spring season and holidays, as people are more likely to spend time with family. Offering more services on working days, when demand is higher for commuting, could improve overall business performance.
