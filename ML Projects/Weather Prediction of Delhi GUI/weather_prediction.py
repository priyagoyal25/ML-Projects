import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly

data = pd.read_csv("DailyDelhiClimateTrain.csv")
st.title("Weather Prediction")


figure = px.line(data, x="date", 
                 y="meantemp", 
                 title='Mean Temperature in Delhi Over the Years')

st.plotly_chart(figure)

figure = px.line(data, x="date", 
                 y="humidity", 
                 title='Humidity in Delhi Over the Years')
st.plotly_chart(figure)

figure = px.line(data, x="date", 
                 y="wind_speed", 
                 title='Wind Speed in Delhi Over the Years')
st.plotly_chart(figure)

figure = px.scatter(data_frame = data, x="humidity",
                    y="meantemp", size="meantemp", 
                    trendline="ols", 
                    title = "Relationship Between Temperature and Humidity")
st.plotly_chart(figure)

data["date"] = pd.to_datetime(data["date"], format = '%Y-%m-%d')
data['year'] = data['date'].dt.year
data["month"] = data["date"].dt.month

plt.style.use('fivethirtyeight')
plt.figure(figsize=(15, 10))
plt.title("Temperature Change in Delhi Over the Years")
sns.lineplot(data = data, x='month', y='meantemp', hue='year')
st.set_option('deprecation.showPyplotGlobalUse', False)
st.pyplot()

forecast_data = data.rename(columns = {"date": "ds", 
                                       "meantemp": "y"})


model = Prophet()
model.fit(forecast_data)
forecasts = model.make_future_dataframe(periods=(pd.to_datetime('2023-12-31') - pd.to_datetime('2017-01-01')).days + 1)
# forecasts = model.make_future_dataframe({'ds': pd.date_range(start='2017-01-01', end='2023-12-31', freq='D')})
# forecasts = model.make_future_dataframe(periods=365)
predictions = model.predict(forecasts)
fig_forecast = plot_plotly(model, predictions)
st.subheader("Forecasted data till 2023 ")
st.plotly_chart(fig_forecast)
# st.write(predictions[['ds', 'yhat']])

date = st.date_input("Enter date")
forecast_for_date = predictions[predictions['ds'] == pd.to_datetime(date)].reset_index(drop=True)

if not forecast_for_date.empty:
    st.write(f"Forecasted Data for {date}")
    forecast_for_date_display = forecast_for_date.rename(columns={'ds': 'Date', 'yhat': 'Forecasted Temperature'})
    st.write(forecast_for_date_display[['Date','Forecasted Temperature']])
else:
    st.write("No forecast available for the selected date.")

# add a button to view humidity or other 
if st.button("View Humidity"):
    forecast_data = data.rename(columns = {"date": "ds", 
                                       "humidity": "y"})
    model=Prophet()
    model.fit(forecast_data)
    forecasts = model.make_future_dataframe(periods=(pd.to_datetime('2023-12-31') - pd.to_datetime('2017-01-01')).days + 1)
    predictions = model.predict(forecasts)
    fig_forecast = plot_plotly(model, predictions)
    st.subheader("Forecasted data till 2023 ")
    st.plotly_chart(fig_forecast)
 
    forecast_for_date = predictions[predictions['ds'] == pd.to_datetime(date)].reset_index(drop=True)
    if not forecast_for_date.empty:
        st.write(f"Forecasted Data for {date}")
        forecast_for_date_display = forecast_for_date.rename(columns={'ds': 'Date', 'yhat': 'Forecasted Humidity'})
        st.write(forecast_for_date_display[['Date','Forecasted Humidity']])
    else:
        st.write("No forecast available for the selected date.")



