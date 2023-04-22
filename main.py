import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf

# Modelling and forecasting
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from skforecast.model_selection import grid_search_forecaster
from skforecast.model_selection import backtesting_forecaster
from skforecast.model_selection_statsmodels import backtesting_sarimax
from skforecast.model_selection_statsmodels import grid_search_sarimax
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

st.title("Website traffic forecaster")

option = st.selectbox("Choose one option:", ("EDA", "Auto regression", "ARIMA"))
data = pd.read_csv("data.csv")
data['date'] = pd.to_datetime(data['date'], format='%d/%m/%y')
data = data.set_index('date')
data = data.asfreq('1D')
data = data.sort_index()
# Split data into train, validation and test
train = data.loc['2020-07-01':'2021-03-01']
validation = data.loc['2021-03-02':'2021-06-01']
test = data.loc['2021-06-02':'2021-08-22']
# st.write("Train shape: ", train.shape)

def generate_monthly_plot():
    fig, ax = plt.subplots(figsize=(9, 4))
    data['month'] = data.index.month
    sns.violinplot(x='month', y='users', data=data)
    return fig

def generate_weekly_plot():
    fig, ax = plt.subplots(figsize=(9, 4))
    data['week'] = data.index.dayofweek
    sns.violinplot(x='week', y='users', data=data)
    return fig
    

if (option == "EDA"):
    st.write("You selected EDA")
    st.write("Data shape: ", data.shape)
    st.write(data.head())

    st.subheader("Data visualization")
    fig, ax = plt.subplots(figsize=(9, 4))
    train['users'].plot(ax=ax, label='train', linewidth=1)
    validation['users'].plot(ax=ax, label='validation', linewidth=1)
    test['users'].plot(ax=ax, label='test', linewidth=1)
    ax.legend();
    st.pyplot(fig)

    st.subheader("Monthly data visualization")
    st.pyplot(generate_monthly_plot())
    st.subheader("Conclusion:")
    st.write("We observed a seasonal pattern with peak traffic in the month of November and December.")
    st.subheader("Weekly data visualization")
    st.pyplot(generate_weekly_plot())
    st.subheader("Conclusion:")
    st.write("We observed a weekly pattern with lower traffic on weekends i.e Saturday and Sunday.")
    st.subheader("Auto correlation plot")
    st.caption("Auto correlation plot shows the correlation between the time series with a lagged version of itself.")
    fig, ax = plt.subplots(figsize=(9, 4))
    plot_acf(train['users'], ax=ax)
    st.pyplot(fig)
    st.subheader("Partial auto correlation plot")
    st.caption("Partial auto correlation plot shows the correlation between the time series with a lagged version of itself but after eliminating the variations already explained by the intervening comparisons.")
    fig, ax = plt.subplots(figsize=(9, 4))
    plot_pacf(train['users'], ax=ax)
    st.pyplot(fig)
    st.subheader("Conclusion:")
    st.write("The auto correlation plot shows that the time series is not stationary, and there is an overall negative correlation between the time series and its lagged version. ")

elif (option == "Auto regression"):
    # st.write(data)
    st.write("You selected Auto regression")
    st.caption("Auto regression is a statistical method for forecasting time series data. It is based on the idea that the output variable depends linearly on its own previous values")
    forecaster = ForecasterAutoreg(regressor=Ridge(random_state=123), lags=12, transformer_y=StandardScaler())
    # forecaster.fit(y=train['users'])
    code = '''#using ridge as regressor
forecaster = ForecasterAutoreg(regressor=Ridge(random_state=123), lags=12, transformer_y=StandardScaler())
forecaster.fit(y=train['users'])
forecaster'''
    # st.write(train['users'])
    st.code(code, language='python')
    st.code(forecaster)
    st.caption("The code above shows how we used ridge as regressor and lags as 12. Lags are the number of previous values to be used as predictors. We also used StandardScaler to transform the target variable.")
    st.write("Backtesting")
    st.caption("Backtesting is a method of testing a predictive model by training it on a subset of the data and testing it on the remaining data. It is used to evaluate how the model will generalize to an independent data set.")
    metric, scores = backtesting_forecaster(forecaster=forecaster, 
                                        y=data['users'], 
                                        initial_train_size=len(train+validation), 
                                        steps=7, 
                                        refit=False, 
                                        fixed_train_size=False, 
                                        metric=mean_absolute_error, 
                                        verbose=True)
    code = '''#backtesting
metric, scores = backtesting_forecaster(forecaster=forecaster, 
                                        y=data['users'], 
                                        initial_train_size=len(train+validation), 
                                        steps=7, 
                                        refit=False, 
                                        fixed_train_size=False, 
                                        metric=mean_absolute_error, 
                                        verbose=True)'''
    st.code(code, language='python')
    st.write("Mean absolute error: ", metric)
    st.write("Predicted: ", scores.head())
    st.subheader("Test vs predicted")
    fig, ax = plt.subplots(figsize=(9, 4))
    test['users'].plot(ax=ax, label='test', linewidth=1)
    scores.plot(ax=ax, label='predicted', linewidth=1)
    ax.legend();
    st.pyplot(fig)
    st.subheader("Tuning hyperparameters")
    st.caption("Hyperparameters are parameters that are not directly learnt within estimators. We will tune the hyper-parameters using grid search. We can tune the hyperparameters to get the best model.")
    forecaster = ForecasterAutoreg(regressor=Ridge(random_state=123), lags=12, transformer_y=StandardScaler())

    param_grid = {'alpha': np.logspace(-3, 3, 10)}
    lags_grid = [7, 14, 21]
    grid_results = grid_search_forecaster(forecaster=forecaster, y = train['users'].append(validation['users']),
                                    param_grid=param_grid, 
                                    lags_grid=lags_grid,
                                    metric=mean_absolute_error,
                                    refit=False, 
                                    initial_train_size=len(train), 
                                    steps=7, 
                                    fixed_train_size=False, 
                                    return_best=True, 
                                    verbose=False)
    code = '''#tuning hyperparameters
forecaster = ForecasterAutoreg(regressor=Ridge(), lags=12, transformer_y=StandardScaler())

param_grid = {'alpha': np.logspace(-3, 3, 10)}
lags_grid = [7, 12, 14, 21]
grid_results = grid_search_forecaster(forecaster=forecaster, y = train['users'].append(validation['users']),
                                    param_grid=param_grid, 
                                    lags_grid=lags_grid,
                                    metric=mean_absolute_error,
                                    refit=False, 
                                    initial_train_size=len(train), 
                                    steps=7, 
                                    fixed_train_size=False, 
                                    return_best=True, 
                                    verbose=False)    '''
    st.code(code, language='python')
    st.write(grid_results.head(1))
    metric, predictions = backtesting_forecaster(
                          forecaster         = forecaster,
                          y                  = data.users,
                          initial_train_size = len(train['users'].append(validation['users'])),
                          steps              = 7,
                          refit              = False,
                          fixed_train_size   = False,
                          metric             = 'mean_absolute_error',
                          verbose            = True
                      )
    

    print(f'Backtest error using test data: {metric}')
    code = '''# Backtest final model using test data
    metric, predictions = backtesting_forecaster(
                            forecaster         = forecaster,
                            y                  = data.users,
                            initial_train_size = len(train['users'].append(validation['users'])),
                            steps              = 7,
                            refit              = False,
                            fixed_train_size   = False,
                            metric             = 'mean_absolute_error',
                            verbose            = True
                        )

    metric'''
    st.code(code, language='python')
    st.write("Backtest error using test data: ", metric)


    
elif (option == "ARIMA"):
    st.write("You selected ARIMA")
    st.caption("ARIMA is a class of models that captures a suite of standard temporal structures in time series data.")
    forecaster = ForecasterAutoreg(regressor=Ridge(random_state=123), lags=12, transformer_y=StandardScaler())
    forecaster.fit(y=train['users'])

    forecaster = ForecasterAutoreg(regressor=Ridge(random_state=123), lags=12, transformer_y=StandardScaler())
    forecaster.fit(y=train['users'])
    st.subheader("Backtest ARIMA")
    metric, predictions = backtesting_sarimax( y = data.users,
    order = (14, 0, 0), initial_train_size = len(train['users'].append(validation['users'])),steps = 7, refit = False, fixed_train_size = False ,metric = 'mean_absolute_error', verbose = True)

    print(f'Backtest error: {metric}')
    predictions.head(5)
    
    code = '''# Backtest using ARIMA
    metric, predictions = backtesting_sarimax( y = data.users,
    order = (14, 0, 0), initial_train_size = len(train['users'].append(validation['users'])),steps = 7, refit = False, fixed_train_size = False ,metric = 'mean_absolute_error', verbose = True)

    print(f'Backtest error: {metric}')
    predictions.head(5)'''
    st.code(code, language='python')
    st.write("Backtest error: ", metric)
    st.write("Predicted: ", predictions.head())
    st.subheader("Test vs predicted")
    fig, ax = plt.subplots(figsize=(9, 4))
    test['users'].plot(ax=ax, label='test', linewidth=1)
    predictions.plot(ax=ax, label='predicted', linewidth=1)
    ax.legend();

    st.pyplot(fig)

    st.subheader("Tuning hyperparameters")
    code = '''#tuning hyperparameters
    param_grid = {'order': [(14, 0, 0), (14, 2, 0), (14, 1, 0), (14, 1, 1),
                            (21, 0, 0), (21, 0, 0), (21, 1, 0), (21, 1, 1)]}
    grid_results = grid_search_sarimax(y = data.users, 
                                        param_grid = param_grid, 
                                        initial_train_size = len(train['users'].append(validation['users'])), 
                                        steps = 7, 
                                        refit = False, 
                                        fixed_train_size = False, 
                                        verbose = False, 
                                        fit_kwargs = {'maxiter': 200, 'disp': 0}, 
                                        metric = 'mean_absolute_error')
    '''
    st.code(code, language='python')

    param_grid = {'order': [(14, 0, 0), (14, 2, 0), (14, 1, 0), (14, 1, 1),
                        (21, 0, 0), (21, 0, 0), (21, 1, 0), (21, 1, 1)]}

    grid_results = grid_search_sarimax(y = data.users, param_grid = param_grid, initial_train_size = len(train['users'].append(validation['users'])), steps = 7, refit = False, fixed_train_size = False, verbose = False, fit_kwargs = {'maxiter': 200, 'disp': 0}, metric = 'mean_absolute_error')

    
    st.write(grid_results.head(1))

    st.subheader("Backtest final model using test data")
    metric, predictions = backtesting_sarimax( y = data.users,order = (12, 1, 1), initial_train_size = len(train['users'].append(validation['users'])), steps = 7, refit = False, fixed_train_size = False ,metric = 'mean_absolute_error', verbose = True)
    print(f'Backtest error: {metric}')
    code = '''# Backtest final model using test data
    metric, predictions = backtesting_sarimax( y = data.users,order = (12, 1, 1), initial_train_size = len(train['users'].append(validation['users'])), steps = 7, refit = False, fixed_train_size = False ,metric = 'mean_absolute_error', verbose = True)
    print(f'Backtest error: {metric}')'''
    st.code(code, language='python')
    st.write("Backtest error: ", metric)
    st.write("Predicted: ", predictions.head())
