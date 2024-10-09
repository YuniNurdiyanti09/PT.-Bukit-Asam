import numpy as np
import pandas as pd
from prophet import Prophet
import warnings
warnings.filterwarnings('ignore')
import itertools
import pyodbc
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_percentage_error as smape
import os
# from dotenv.main import load_dotenv

# load_dotenv()
# Koneksi ke SQL Server
server = '10.3.4.139,1433'
database = 'dwh_prod'
username = 'developerptba'
password = 'LnfPYVFW1K2TAKf'
# server = os.environ['SERVER']
# database = os.environ['DATABASE']
# username = os.environ['USERNAME_1']
# password = os.environ['PASSWORD']

cnxn = pyodbc.connect('DRIVER={SQL Server};SERVER='+server+';DATABASE='+database+';UID='+username+';PWD='+ password)
cursor = cnxn.cursor()
query = "SELECT * FROM dwh.DM_financial_position_unpivot_full;"
df = pd.read_sql(query, cnxn)

# filter pada Dashboard Overview (dibuat dalam list)
BD_Overview =['Total Production',
    'BAP',
    'Total Transport',
    'Total Qty of Coal Sales',
    'SR',
    'Coal Cash Cost',
    'Average Market Price IDR',
    'Cash and Deposit',
    'Total Revenue',
    'EBITDA After Minority',
    'Net Profit After Minority'
]

# filter data berdasrkan Breakdown Yang dipilih
df_overview_filter = df[df['BreakDown'].isin(BD_Overview)]


# sum total production dari BD = total Production + BAP
df_overview_filter['BreakDown2']= df_overview_filter['BreakDown'].apply(lambda x: 'Total Production' if x in ['Total Production','BAP'] else x)


# Filter data yag akan dilakukan model dengan jenis : Actual
# df_overview_actual : jenis : aktual
df_overview_actual = df_overview_filter[df_overview_filter['tipe'] == "Actual"]

# data diurutkan dari awal data
df_overview_date = df_overview_actual.sort_values(by='Date')


df_overview = df_overview_date.groupby(['Date','BreakDown2'])['amount'].sum().reset_index()


# PRODUCTION
print("--"*60)
print("PRODUCTION")

df_production = df_overview[df_overview['BreakDown2']=='Total Production'][['Date','amount']]
# Reset index
df_production = df_production.reset_index(drop=True)
df_production.columns = ['ds','y']
df_production['ds'] = pd.to_datetime(df_production['ds'])

# Kombinasi parameter untuk tuning
changepoint_prior_scales = [0.01, 0.02]
seasonality_modes = ['additive']
seasonality_prior_scales = [0.01, 0.05, 1.0]

# Membuat list kombinasi parameter
param_combinations = list(itertools.product(changepoint_prior_scales, seasonality_modes, seasonality_prior_scales))

# Fungsi untuk menghitung SMAPE
def smape(actual, predicted):
    return 100 * np.mean(2 * np.abs(predicted - actual) / (np.abs(predicted) + np.abs(actual)))

# Menyimpan hasil tuning
best_smape = float('inf')
best_params = None

# Train dan test split
train_size = int(0.8 * len(df_production))
train_data = df_production[:train_size]
test_data = df_production[train_size:]

# Looping untuk setiap kombinasi parameter
for changepoint_prior_scale, seasonality_mode, seasonality_prior_scale in param_combinations:
    
    # Inisialisasi model Prophet dengan parameter tuning
    model = Prophet(
        yearly_seasonality=True, 
        weekly_seasonality=False, 
        daily_seasonality=False,
        changepoint_prior_scale=changepoint_prior_scale,
        seasonality_mode=seasonality_mode,
        seasonality_prior_scale=seasonality_prior_scale
    )
    model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
    
    # Melatih model dengan data train
    model.fit(train_data)

    # Membuat dataframe masa depan yang mencakup periode data test
    future = model.make_future_dataframe(periods=len(test_data), freq='MS')

    # Melakukan prediksi
    forecast = model.predict(future)

    # Mengambil prediksi yang sesuai dengan data test
    forecast_test = forecast[-len(test_data):]

    # Menggabungkan data asli dengan prediksi
    test_data.loc[:, 'yhat'] = forecast_test['yhat'].values

    # Menghitung SMAPE untuk data test
    smape_value = smape(test_data['y'], test_data['yhat'])
    print(f"SMAPE dengan changepoint_prior_scale={changepoint_prior_scale}, seasonality_mode={seasonality_mode}, seasonality_prior_scale={seasonality_prior_scale}: {smape_value:.2f}%")

    # Menyimpan kombinasi parameter terbaik
    if smape_value < best_smape:
        best_smape = smape_value
        best_params = (changepoint_prior_scale, seasonality_mode, seasonality_prior_scale)

print(f"\nParameter terbaik: changepoint_prior_scale={best_params[0]}, seasonality_mode={best_params[1]}, seasonality_prior_scale={best_params[2]}")
print(f"SMAPE terbaik: {best_smape:.2f}%")

# Latih model dengan seluruh data menggunakan parameter terbaik
final_model = Prophet(
    yearly_seasonality=True, 
    weekly_seasonality=False, 
    daily_seasonality=False,
    changepoint_prior_scale=best_params[0],
    seasonality_mode=best_params[1],
    seasonality_prior_scale=best_params[2]
)
final_model.add_seasonality(name='monthly', period=30.5, fourier_order=5)

# Melatih model dengan seluruh data
final_model.fit(df_production)

# Membuat dataframe masa depan untuk 6 bulan ke depan
future = final_model.make_future_dataframe(periods=12, freq='MS')

# Melakukan prediksi
forecast_final = final_model.predict(future)

# Menampilkan hasil prediksi
print(forecast_final[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])

# Menyimpan hasil prediksi ke dalam DataFrame
forecast_df = forecast_final[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
forecast_df.columns = ['Date', 'Forecast', 'Lower Bound', 'Upper Bound']

# Membulatkan kolom Forecast, Lower Bound, dan Upper Bound ke dua angka di belakang koma
forecast_df['Forecast'] = forecast_df['Forecast'].round(2)
forecast_df['Lower Bound'] = forecast_df['Lower Bound'].round(2)
forecast_df['Upper Bound'] = forecast_df['Upper Bound'].round(2)

# Menambahkan data untuk 'Forecast'
forecast_rows = pd.DataFrame({
    'ds': forecast_df['Date'],
    'y': forecast_df['Forecast'],
    'jenis': 'forecast',
    'Breakdown' : 'Total Production'
})

# Menambahkan data untuk 'Lower Bound'
lower_bound_rows = pd.DataFrame({
    'ds': forecast_df['Date'],
    'y': forecast_df['Lower Bound'],
    'jenis': 'lower bound',
    'Breakdown' : 'Total Production'
})

# Menambahkan data untuk 'Upper Bound'
upper_bound_rows = pd.DataFrame({
    'ds': forecast_df['Date'],
    'y': forecast_df['Upper Bound'],
    'jenis': 'upper bound',
    'Breakdown' : 'Total Production'
})
df_production_actual = pd.DataFrame({
    'ds': df_production['ds'],
    'y': df_production['y'],
    'jenis': 'actual',
    'Breakdown' : 'Total Production'
})
# Menggabungkan semua DataFrame
result_df_production = pd.concat([df_production_actual,forecast_rows, lower_bound_rows, upper_bound_rows], ignore_index=True)

# SR
print("--"*60)
print("SR")

df_SR = df_overview[df_overview['BreakDown2']=='SR'][['Date','amount']]
# Reset index
df_SR = df_SR.reset_index(drop=True)
df_SR.columns = ['ds','y']
df_SR['ds'] = pd.to_datetime(df_SR['ds'])

# Kombinasi parameter untuk tuning
changepoint_prior_scales = [0.05, 0.1]
seasonality_modes = ['additive']
seasonality_prior_scales = [0.01, 0.05, 1.0]

# Membuat list kombinasi parameter
param_combinations = list(itertools.product(changepoint_prior_scales, seasonality_modes, seasonality_prior_scales))

# Fungsi untuk menghitung SMAPE
def smape(actual, predicted):
    return 100 * np.mean(2 * np.abs(predicted - actual) / (np.abs(predicted) + np.abs(actual)))

# Menyimpan hasil tuning
best_smape = float('inf')
best_params = None

# Train dan test split
train_size = int(0.8 * len(df_SR))
train_data = df_SR[:train_size]
test_data = df_SR[train_size:]

# Looping untuk setiap kombinasi parameter
for changepoint_prior_scale, seasonality_mode, seasonality_prior_scale in param_combinations:
    
    # Inisialisasi model Prophet dengan parameter tuning
    model = Prophet(
        yearly_seasonality=True, 
        weekly_seasonality=False, 
        daily_seasonality=False,
        changepoint_prior_scale=changepoint_prior_scale,
        seasonality_mode=seasonality_mode,
        seasonality_prior_scale=seasonality_prior_scale
    )
    model.add_seasonality(name='monthly', period=30.5, fourier_order=3)
    
    # Melatih model dengan data train
    model.fit(train_data)

    # Membuat dataframe masa depan yang mencakup periode data test
    future = model.make_future_dataframe(periods=len(test_data), freq='MS')

    # Melakukan prediksi
    forecast = model.predict(future)

    # Mengambil prediksi yang sesuai dengan data test
    forecast_test = forecast[-len(test_data):]

    # Menggabungkan data asli dengan prediksi
    test_data.loc[:, 'yhat'] = forecast_test['yhat'].values

    # Menghitung SMAPE untuk data test
    smape_value = smape(test_data['y'], test_data['yhat'])
    print(f"SMAPE dengan changepoint_prior_scale={changepoint_prior_scale}, seasonality_mode={seasonality_mode}, seasonality_prior_scale={seasonality_prior_scale}: {smape_value:.2f}%")

    # Menyimpan kombinasi parameter terbaik
    if smape_value < best_smape:
        best_smape = smape_value
        best_params = (changepoint_prior_scale, seasonality_mode, seasonality_prior_scale)

print(f"\nParameter terbaik: changepoint_prior_scale={best_params[0]}, seasonality_mode={best_params[1]}, seasonality_prior_scale={best_params[2]}")
print(f"SMAPE terbaik: {best_smape:.2f}%")

# Latih model dengan seluruh data menggunakan parameter terbaik
final_model = Prophet(
    yearly_seasonality=True, 
    weekly_seasonality=False, 
    daily_seasonality=False,
    changepoint_prior_scale=best_params[0],
    seasonality_mode=best_params[1],
    seasonality_prior_scale=best_params[2]
)
final_model.add_seasonality(name='monthly', period=30.5, fourier_order=5)

# Melatih model dengan seluruh data
final_model.fit(df_SR)

# Membuat dataframe masa depan untuk 6 bulan ke depan
future = final_model.make_future_dataframe(periods=12, freq='MS')

# Melakukan prediksi
forecast_final = final_model.predict(future)

# Menampilkan hasil prediksi
print(forecast_final[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])

# Menyimpan hasil prediksi ke dalam DataFrame
forecast_df_SR = forecast_final[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
forecast_df_SR.columns = ['Date', 'Forecast', 'Lower Bound', 'Upper Bound']

forecast_df_SR['Forecast'] = forecast_df_SR['Forecast'].round(2)
forecast_df_SR['Lower Bound'] = forecast_df_SR['Lower Bound'].round(2)
forecast_df_SR['Upper Bound'] = forecast_df_SR['Upper Bound'].round(2)

# Memasukkan kedalam dataframe
forecast_df_SR.loc[:, 'Date'] = pd.to_datetime(forecast_df_SR['Date'])
# Menambahkan data untuk 'Forecast'
forecast_rows = pd.DataFrame({
    'ds': forecast_df_SR['Date'],
    'y': forecast_df_SR['Forecast'],
    'jenis': 'forecast',
    'Breakdown' : 'SR'
})

# Menambahkan data untuk 'Lower Bound'
lower_bound_rows = pd.DataFrame({
    'ds': forecast_df_SR['Date'],
    'y': forecast_df_SR['Lower Bound'],
    'jenis': 'lower bound',
    'Breakdown' : 'SR'
})

# Menambahkan data untuk 'Upper Bound'
upper_bound_rows = pd.DataFrame({
    'ds': forecast_df_SR['Date'],
    'y': forecast_df_SR['Upper Bound'],
    'jenis': 'upper bound',
    'Breakdown' : 'SR'
})
df_SR_actual = pd.DataFrame({
    'ds': df_SR['ds'],
    'y': df_SR['y'],
    'jenis': 'actual',
    'Breakdown' : 'SR'
})
# Menggabungkan semua DataFrame
result_df_SR = pd.concat([df_SR_actual,forecast_rows, lower_bound_rows, upper_bound_rows], ignore_index=True)

# Transport
print("--"*60)
print("TRANSPORT")

df_transport = df_overview[df_overview['BreakDown2']=='Total Transport'][['Date','amount']]
# Reset index
df_transport = df_transport.reset_index(drop=True)
df_transport.columns = ['ds','y']
df_transport['ds'] = pd.to_datetime(df_transport['ds'])

# Kombinasi parameter untuk tuning SARIMA
p = d = q = range(0, 2)  # AR, I, MA order
P = D = Q = range(0, 2)  # Seasonal ARIMA order
s = [12]  # Musiman dalam bulan (12 bulan)

# Membuat list kombinasi parameter p, d, q, P, D, Q, s
param_combinations = list(itertools.product(p, d, q, P, D, Q, s))

# Menyimpan hasil tuning
best_smape = float('inf')
best_params = None

# Train dan test split
train_size = int(0.8 * len(df_transport))
train_data = df_transport[:train_size]
test_data = df_transport[train_size:]

# Fungsi SMAPE
def smape(actual, predicted):
    return 100 * np.mean(2 * np.abs(predicted - actual) / (np.abs(predicted) + np.abs(actual)))

# Looping untuk setiap kombinasi parameter
for (p, d, q, P, D, Q, s) in param_combinations:
    try:
        # Inisialisasi model SARIMA dengan parameter tuning
        model = SARIMAX(train_data['y'],
                        order=(p, d, q),
                        seasonal_order=(P, D, Q, s),
                        enforce_stationarity=False,
                        enforce_invertibility=False)

        # Melatih model dengan data train
        model_fit = model.fit(disp=False)

        # Melakukan prediksi
        forecast = model_fit.forecast(steps=len(test_data))

        # Menggabungkan data asli dengan prediksi
        test_data.loc[:, 'yhat'] = forecast

        # Menghitung SMAPE untuk data test
        smape_value = smape(test_data['y'], test_data['yhat'])
        print(f"SMAPE dengan order=({p},{d},{q}), seasonal_order=({P},{D},{Q},{s}): {smape_value:.2f}%")

        # Menyimpan kombinasi parameter terbaik
        if smape_value < best_smape:
            best_smape = smape_value
            best_params = (p, d, q, P, D, Q, s)

    except Exception as e:
        print(f"Error untuk parameter {p, d, q, P, D, Q, s}: {e}")

print(f"\nParameter terbaik: order=({best_params[0]},{best_params[1]},{best_params[2]}), seasonal_order=({best_params[3]},{best_params[4]},{best_params[5]},{best_params[6]})")
print(f"SMAPE terbaik: {best_smape:.2f}%")

# Membuat model SARIMAX menggunakan parameter terbaik
final_model = SARIMAX(df_transport['y'],
                     order=(best_params[0], best_params[1], best_params[2]),
                     seasonal_order=(best_params[3], best_params[4], best_params[5], best_params[6]),
                     enforce_stationarity=False,
                     enforce_invertibility=False)

# Melatih model dengan seluruh data
final_model_fit = final_model.fit(disp=False)

# Prediksi untuk seluruh data historis (dari awal data hingga akhir)
forecast_historic = final_model_fit.get_prediction(steps=len(df_transport))
forecast_historic_mean = forecast_historic.predicted_mean
forecast_historic_ci = forecast_historic.conf_int()

# Prediksi untuk 12 bulan ke depan
forecast_future = final_model_fit.get_forecast(steps=12)
forecast_future_mean = forecast_future.predicted_mean
forecast_future_ci = forecast_future.conf_int()

# Membuat tanggal untuk prediksi masa depan
future_dates = pd.date_range(start=df_transport['ds'].max() + pd.DateOffset(months=1), periods=12, freq='MS')

# Konversi 'future_dates' ke Series agar bisa digabungkan dengan tanggal historis
future_dates_series = pd.Series(future_dates)

# Menyimpan hasil prediksi ke dalam DataFrame untuk data historis dan data masa depan
forecast_df_transport = pd.DataFrame({
    'Date': pd.concat([df_transport['ds'].reset_index(drop=True), future_dates_series], ignore_index=True),
    'Forecast': pd.concat([forecast_historic_mean, forecast_future_mean], ignore_index=True),
    'Lower Bound': pd.concat([forecast_historic_ci.iloc[:, 0], forecast_future_ci.iloc[:, 0]], ignore_index=True),
    'Upper Bound': pd.concat([forecast_historic_ci.iloc[:, 1], forecast_future_ci.iloc[:, 1]], ignore_index=True)
})

forecast_df_transport['Forecast'] = forecast_df_transport['Forecast'].round(2)
forecast_df_transport['Lower Bound'] = forecast_df_transport['Lower Bound'].round(2)
forecast_df_transport['Upper Bound'] = forecast_df_transport['Upper Bound'].round(2)
# Membuat DF Transport
forecast_df_transport.loc[:, 'Date'] = pd.to_datetime(forecast_df_transport['Date'])
# Menambahkan data untuk 'Forecast'
forecast_rows = pd.DataFrame({
    'ds': forecast_df_transport['Date'],
    'y': forecast_df_transport['Forecast'],
    'jenis': 'forecast',
    'Breakdown' : 'Total Transport'
})

# Menambahkan data untuk 'Lower Bound'
lower_bound_rows = pd.DataFrame({
    'ds': forecast_df_transport['Date'],
    'y': forecast_df_transport['Lower Bound'],
    'jenis': 'lower bound',
    'Breakdown' : 'Total Transport'
})

# Menambahkan data untuk 'Upper Bound'
upper_bound_rows = pd.DataFrame({
    'ds': forecast_df_transport['Date'],
    'y': forecast_df_transport['Upper Bound'],
    'jenis': 'upper bound',
    'Breakdown' : 'Total Transport'
})
df_Transport_actual = pd.DataFrame({
    'ds': df_transport['ds'],
    'y': df_transport['y'],
    'jenis': 'actual',
    'Breakdown' : 'Total Transport'
})
# Menggabungkan semua DataFrame
result_df_Transport = pd.concat([df_Transport_actual,forecast_rows, lower_bound_rows, upper_bound_rows], ignore_index=True)

# SALES
print("--"*60)
print("SALES")

df_Sales = df_overview[df_overview['BreakDown2']=='Total Qty of Coal Sales'][['Date','amount']]
# Reset index
df_Sales = df_Sales.reset_index(drop=True)
df_Sales.columns = ['ds','y']
df_Sales['ds'] = pd.to_datetime(df_Sales['ds'])


#  Kombinasi parameter untuk tuning SARIMA
p = d = q = range(0, 2)  # AR, I, MA order
P = D = Q = range(0, 2)  # Seasonal ARIMA order
s = [12]  # Musiman dalam bulan (12 bulan)

# Membuat list kombinasi parameter p, d, q, P, D, Q, s
param_combinations = list(itertools.product(p, d, q, P, D, Q, s))

# Menyimpan hasil tuning
best_smape = float('inf')
best_params = None

# Train dan test split
train_size = int(0.8 * len(df_Sales))
train_data = df_Sales[:train_size]
test_data = df_Sales[train_size:]

# Fungsi SMAPE
def smape(actual, predicted):
    return 100 * np.mean(2 * np.abs(predicted - actual) / (np.abs(predicted) + np.abs(actual)))

# Looping untuk setiap kombinasi parameter
for (p, d, q, P, D, Q, s) in param_combinations:
    try:
        # Inisialisasi model SARIMA dengan parameter tuning
        model = SARIMAX(train_data['y'],
                        order=(p, d, q),
                        seasonal_order=(P, D, Q, s),
                        enforce_stationarity=False,
                        enforce_invertibility=False)

        # Melatih model dengan data train
        model_fit = model.fit(disp=False)

        # Melakukan prediksi
        forecast = model_fit.forecast(steps=len(test_data))

        # Menggabungkan data asli dengan prediksi
        test_data.loc[:, 'yhat'] = forecast

        # Menghitung SMAPE untuk data test
        smape_value = smape(test_data['y'], test_data['yhat'])
        print(f"SMAPE dengan order=({p},{d},{q}), seasonal_order=({P},{D},{Q},{s}): {smape_value:.2f}%")

        # Menyimpan kombinasi parameter terbaik
        if smape_value < best_smape:
            best_smape = smape_value
            best_params = (p, d, q, P, D, Q, s)

    except Exception as e:
        print(f"Error untuk parameter {p, d, q, P, D, Q, s}: {e}")

print(f"\nParameter terbaik: order=({best_params[0]},{best_params[1]},{best_params[2]}), seasonal_order=({best_params[3]},{best_params[4]},{best_params[5]},{best_params[6]})")
print(f"SMAPE terbaik: {best_smape:.2f}%")

# Membuat model SARIMAX menggunakan parameter terbaik
final_model = SARIMAX(df_Sales['y'],
                     order=(best_params[0], best_params[1], best_params[2]),
                     seasonal_order=(best_params[3], best_params[4], best_params[5], best_params[6]),
                     enforce_stationarity=False,
                     enforce_invertibility=False)

# Melatih model dengan seluruh data
final_model_fit = final_model.fit(disp=False)

# Prediksi untuk seluruh data historis (dari awal data hingga akhir)
forecast_historic = final_model_fit.get_prediction(steps=len(df_Sales))
forecast_historic_mean = forecast_historic.predicted_mean
forecast_historic_ci = forecast_historic.conf_int()

# Prediksi untuk 12 bulan ke depan
forecast_future = final_model_fit.get_forecast(steps=12)
forecast_future_mean = forecast_future.predicted_mean
forecast_future_ci = forecast_future.conf_int()

# Membuat tanggal untuk prediksi masa depan
future_dates = pd.date_range(start=df_Sales['ds'].max() + pd.DateOffset(months=1), periods=12, freq='MS')

# Konversi 'future_dates' ke Series agar bisa digabungkan dengan tanggal historis
future_dates_series = pd.Series(future_dates)

# Menyimpan hasil prediksi ke dalam DataFrame untuk data historis dan data masa depan
forecast_df_Sales = pd.DataFrame({
    'Date': pd.concat([df_Sales['ds'].reset_index(drop=True), future_dates_series], ignore_index=True),
    'Forecast': pd.concat([forecast_historic_mean, forecast_future_mean], ignore_index=True),
    'Lower Bound': pd.concat([forecast_historic_ci.iloc[:, 0], forecast_future_ci.iloc[:, 0]], ignore_index=True),
    'Upper Bound': pd.concat([forecast_historic_ci.iloc[:, 1], forecast_future_ci.iloc[:, 1]], ignore_index=True)
})

forecast_df_Sales['Forecast'] = forecast_df_Sales['Forecast'].round(2)
forecast_df_Sales['Lower Bound'] = forecast_df_Sales['Lower Bound'].round(2)
forecast_df_Sales['Upper Bound'] = forecast_df_Sales['Upper Bound'].round(2)

# forecast_df_Sales

forecast_df_Sales.loc[:, 'Date'] = pd.to_datetime(forecast_df_Sales['Date'])
# Menambahkan data untuk 'Forecast'
forecast_rows = pd.DataFrame({
    'ds': forecast_df_Sales['Date'],
    'y': forecast_df_Sales['Forecast'],
    'jenis': 'forecast',
    'Breakdown' : 'Total Qty of Coal Sales'
})

# Menambahkan data untuk 'Lower Bound'
lower_bound_rows = pd.DataFrame({
    'ds': forecast_df_Sales['Date'],
    'y': forecast_df_Sales['Lower Bound'],
    'jenis': 'lower bound',
    'Breakdown' : 'Total Qty of Coal Sales'
})

# Menambahkan data untuk 'Upper Bound'
upper_bound_rows = pd.DataFrame({
    'ds': forecast_df_Sales['Date'],
    'y': forecast_df_Sales['Upper Bound'],
    'jenis': 'upper bound',
    'Breakdown' : 'Total Qty of Coal Sales'
})
df_Sales_actual = pd.DataFrame({
    'ds': df_Sales['ds'],
    'y': df_Sales['y'],
    'jenis': 'actual',
    'Breakdown' : 'Total Qty of Coal Sales'
})
# Menggabungkan semua DataFrame
result_df_Sales = pd.concat([df_Sales_actual,forecast_rows, lower_bound_rows, upper_bound_rows], ignore_index=True)

# Cast Cost
print("--"*60)
print("CASH COST")

df_coal_cast_cost = df_overview[df_overview['BreakDown2']=='Coal Cash Cost'][['Date','amount']]
# Reset index
df_coal_cast_cost = df_coal_cast_cost.reset_index(drop=True)
df_coal_cast_cost.columns = ['ds','y']
df_coal_cast_cost['ds'] = pd.to_datetime(df_coal_cast_cost['ds'])

# Kombinasi parameter untuk tuning
changepoint_prior_scales = [0.05, 0.1, 0.2]
seasonality_modes = ['additive']
seasonality_prior_scales = [0.1, 0.2]

# Membuat list kombinasi parameter
param_combinations = list(itertools.product(changepoint_prior_scales, seasonality_modes, seasonality_prior_scales))

# Fungsi untuk menghitung SMAPE
def smape(actual, predicted):
    return 100 * np.mean(2 * np.abs(predicted - actual) / (np.abs(predicted) + np.abs(actual)))

# Menyimpan hasil tuning
best_smape = float('inf')
best_params = None

# Train dan test split
train_size = int(0.8 * len(df_coal_cast_cost))
train_data = df_coal_cast_cost[:train_size]
test_data = df_coal_cast_cost[train_size:]

# Looping untuk setiap kombinasi parameter
for changepoint_prior_scale, seasonality_mode, seasonality_prior_scale in param_combinations:
    
    # Inisialisasi model Prophet dengan parameter tuning
    model = Prophet(
        yearly_seasonality=True, 
        # weekly_seasonality=False, 
        # daily_seasonality=False,
        changepoint_prior_scale=changepoint_prior_scale,
        seasonality_mode=seasonality_mode,
        seasonality_prior_scale=seasonality_prior_scale
    )
    model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
    
    # Melatih model dengan data train
    model.fit(train_data)

    # Membuat dataframe masa depan yang mencakup periode data test
    future = model.make_future_dataframe(periods=len(test_data), freq='MS')

    # Melakukan prediksi
    forecast = model.predict(future)

    # Mengambil prediksi yang sesuai dengan data test
    forecast_test = forecast[-len(test_data):]

    # Menggabungkan data asli dengan prediksi
    test_data.loc[:, 'yhat'] = forecast_test['yhat'].values

    # Menghitung SMAPE untuk data test
    smape_value = smape(test_data['y'], test_data['yhat'])
    print(f"SMAPE dengan changepoint_prior_scale={changepoint_prior_scale}, seasonality_mode={seasonality_mode}, seasonality_prior_scale={seasonality_prior_scale}: {smape_value:.2f}%")

    # Menyimpan kombinasi parameter terbaik
    if smape_value < best_smape:
        best_smape = smape_value
        best_params = (changepoint_prior_scale, seasonality_mode, seasonality_prior_scale)

print(f"\nParameter terbaik: changepoint_prior_scale={best_params[0]}, seasonality_mode={best_params[1]}, seasonality_prior_scale={best_params[2]}")
print(f"SMAPE terbaik: {best_smape:.2f}%")

# Latih model dengan seluruh data menggunakan parameter terbaik
final_model = Prophet(
    yearly_seasonality=True, 
    # weekly_seasonality=False, 
    # daily_seasonality=False,
    changepoint_prior_scale=best_params[0],
    seasonality_mode=best_params[1],
    seasonality_prior_scale=best_params[2]
)
final_model.add_seasonality(name='monthly', period=30.5, fourier_order=5)

# Melatih model dengan seluruh data
final_model.fit(df_coal_cast_cost)

# Membuat dataframe masa depan untuk 6 bulan ke depan
future = final_model.make_future_dataframe(periods=12, freq='MS')

# Melakukan prediksi
forecast_final = final_model.predict(future)

# Menampilkan hasil prediksi
print(forecast_final[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])

# Menyimpan hasil prediksi ke dalam DataFrame
forecast_df_coal_cast_cost = forecast_final[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
forecast_df_coal_cast_cost.columns = ['Date', 'Forecast', 'Lower Bound', 'Upper Bound']

forecast_df_coal_cast_cost['Forecast'] = forecast_df_coal_cast_cost['Forecast'].round(2)
forecast_df_coal_cast_cost['Lower Bound'] = forecast_df_coal_cast_cost['Lower Bound'].round(2)
forecast_df_coal_cast_cost['Upper Bound'] = forecast_df_coal_cast_cost['Upper Bound'].round(2)

forecast_df_coal_cast_cost.loc[:, 'Date'] = pd.to_datetime(forecast_df_coal_cast_cost['Date'])
# Menambahkan data untuk 'Forecast'
forecast_rows = pd.DataFrame({
    'ds': forecast_df_coal_cast_cost['Date'],
    'y': forecast_df_coal_cast_cost['Forecast'],
    'jenis': 'forecast',
    'Breakdown' : 'Coal Cash Cost'
})

# Menambahkan data untuk 'Lower Bound'
lower_bound_rows = pd.DataFrame({
    'ds': forecast_df_coal_cast_cost['Date'],
    'y': forecast_df_coal_cast_cost['Lower Bound'],
    'jenis': 'lower bound',
    'Breakdown' : 'Coal Cash Cost'
})

# Menambahkan data untuk 'Upper Bound'
upper_bound_rows = pd.DataFrame({
    'ds': forecast_df_coal_cast_cost['Date'],
    'y': forecast_df_coal_cast_cost['Upper Bound'],
    'jenis': 'upper bound',
    'Breakdown' : 'Coal Cash Cost'
})
df_coal_cast_cost_actual = pd.DataFrame({
    'ds': df_coal_cast_cost['ds'],
    'y': df_coal_cast_cost['y'],
    'jenis': 'actual',
    'Breakdown' : 'Coal Cash Cost'
})
# Menggabungkan semua DataFrame
result_df_coal_cast_cost = pd.concat([df_coal_cast_cost_actual,forecast_rows, lower_bound_rows, upper_bound_rows], ignore_index=True)


# AVG Selling Price
print("--"*60)
print("AVG. SELLING PRICE")

df_AvgSP = df_overview[df_overview['BreakDown2']=='Average Market Price IDR'][['Date','amount']]
# Reset index
df_AvgSP = df_AvgSP.reset_index(drop=True)
df_AvgSP.columns = ['ds','y']
df_AvgSP['ds'] = pd.to_datetime(df_AvgSP['ds'])

# Kombinasi parameter untuk tuning
changepoint_prior_scales = [0.05, 0.1, 0.2]
seasonality_modes = ['additive','multiplicative']
seasonality_prior_scales = [0.1, 0.2]

# Membuat list kombinasi parameter
param_combinations = list(itertools.product(changepoint_prior_scales, seasonality_modes, seasonality_prior_scales))

# Fungsi untuk menghitung SMAPE
def smape(actual, predicted):
    return 100 * np.mean(2 * np.abs(predicted - actual) / (np.abs(predicted) + np.abs(actual)))

# Menyimpan hasil tuning
best_smape = float('inf')
best_params = None

# Train dan test split
train_size = int(0.8 * len(df_AvgSP))
train_data = df_AvgSP[:train_size]
test_data = df_AvgSP[train_size:]

# Looping untuk setiap kombinasi parameter
for changepoint_prior_scale, seasonality_mode, seasonality_prior_scale in param_combinations:
    
    # Inisialisasi model Prophet dengan parameter tuning
    model = Prophet(
        yearly_seasonality=True, 
        # weekly_seasonality=False, 
        # daily_seasonality=False,
        changepoint_prior_scale=changepoint_prior_scale,
        seasonality_mode=seasonality_mode,
        seasonality_prior_scale=seasonality_prior_scale
    )
    model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
    
    # Melatih model dengan data train
    model.fit(train_data)

    # Membuat dataframe masa depan yang mencakup periode data test
    future = model.make_future_dataframe(periods=len(test_data), freq='MS')

    # Melakukan prediksi
    forecast = model.predict(future)

    # Mengambil prediksi yang sesuai dengan data test
    forecast_test = forecast[-len(test_data):]

    # Menggabungkan data asli dengan prediksi
    test_data.loc[:, 'yhat'] = forecast_test['yhat'].values

    # Menghitung SMAPE untuk data test
    smape_value = smape(test_data['y'], test_data['yhat'])
    print(f"SMAPE dengan changepoint_prior_scale={changepoint_prior_scale}, seasonality_mode={seasonality_mode}, seasonality_prior_scale={seasonality_prior_scale}: {smape_value:.2f}%")

    # Menyimpan kombinasi parameter terbaik
    if smape_value < best_smape:
        best_smape = smape_value
        best_params = (changepoint_prior_scale, seasonality_mode, seasonality_prior_scale)

print(f"\nParameter terbaik: changepoint_prior_scale={best_params[0]}, seasonality_mode={best_params[1]}, seasonality_prior_scale={best_params[2]}")
print(f"SMAPE terbaik: {best_smape:.2f}%")

# Latih model dengan seluruh data menggunakan parameter terbaik
final_model = Prophet(
    yearly_seasonality=True, 
    # weekly_seasonality=False, 
    # daily_seasonality=False,
    changepoint_prior_scale=best_params[0],
    seasonality_mode=best_params[1],
    seasonality_prior_scale=best_params[2]
)
final_model.add_seasonality(name='monthly', period=30.5, fourier_order=5)

# Melatih model dengan seluruh data
final_model.fit(df_AvgSP)

# Membuat dataframe masa depan untuk 6 bulan ke depan
future = final_model.make_future_dataframe(periods=12, freq='MS')

# Melakukan prediksi
forecast_final = final_model.predict(future)

# Menampilkan hasil prediksi
print(forecast_final[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])

# Menyimpan hasil prediksi ke dalam DataFrame
forecast_df_AvgSP = forecast_final[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
forecast_df_AvgSP.columns = ['Date', 'Forecast', 'Lower Bound', 'Upper Bound']

forecast_df_AvgSP['Forecast'] = forecast_df_AvgSP['Forecast'].round(2)
forecast_df_AvgSP['Lower Bound'] = forecast_df_AvgSP['Lower Bound'].round(2)
forecast_df_AvgSP['Upper Bound'] = forecast_df_AvgSP['Upper Bound'].round(2)

forecast_df_coal_cast_cost.loc[:, 'Date'] = pd.to_datetime(forecast_df_AvgSP['Date'])
# Menambahkan data untuk 'Forecast'
forecast_rows = pd.DataFrame({
    'ds': forecast_df_AvgSP['Date'],
    'y': forecast_df_AvgSP['Forecast'],
    'jenis': 'forecast',
    'Breakdown' : 'Average Market Price IDR'
})

# Menambahkan data untuk 'Lower Bound'
lower_bound_rows = pd.DataFrame({
    'ds': forecast_df_AvgSP['Date'],
    'y': forecast_df_AvgSP['Lower Bound'],
    'jenis': 'lower bound',
    'Breakdown' : 'Average Market Price IDR'
})

# Menambahkan data untuk 'Upper Bound'
upper_bound_rows = pd.DataFrame({
    'ds': forecast_df_AvgSP['Date'],
    'y': forecast_df_coal_cast_cost['Upper Bound'],
    'jenis': 'upper bound',
    'Breakdown' : 'Average Market Price IDR'
})
df_Avg_SP_actual = pd.DataFrame({
    'ds': df_AvgSP['ds'],
    'y': df_AvgSP['y'],
    'jenis': 'actual',
    'Breakdown' : 'Average Market Price IDR'
})
# Menggabungkan semua DataFrame
result_df_Avg_SP = pd.concat([df_Avg_SP_actual,forecast_rows, lower_bound_rows, upper_bound_rows], ignore_index=True)

# Revenue
print("--"*60)
print("REVENUE")

df_Revenue = df_overview[df_overview['BreakDown2']=='Total Revenue'][['Date','amount']]
# Reset index
df_Revenue = df_Revenue.reset_index(drop=True)
df_Revenue.columns = ['ds','y']
df_Revenue['ds'] = pd.to_datetime(df_Revenue['ds'])

# Kombinasi parameter untuk tuning
changepoint_prior_scales = [0.05, 0.1, 0.2]
seasonality_modes = ['additive','multiplicative']
seasonality_prior_scales = [0.1, 0.2]

# Membuat list kombinasi parameter
param_combinations = list(itertools.product(changepoint_prior_scales, seasonality_modes, seasonality_prior_scales))

# Fungsi untuk menghitung SMAPE
def smape(actual, predicted):
    return 100 * np.mean(2 * np.abs(predicted - actual) / (np.abs(predicted) + np.abs(actual)))

# Menyimpan hasil tuning
best_smape = float('inf')
best_params = None

# Train dan test split
train_size = int(0.8 * len(df_Revenue))
train_data = df_Revenue[:train_size]
test_data = df_Revenue[train_size:]

# Looping untuk setiap kombinasi parameter
for changepoint_prior_scale, seasonality_mode, seasonality_prior_scale in param_combinations:
    
    # Inisialisasi model Prophet dengan parameter tuning
    model = Prophet(
        yearly_seasonality=True, 
        # weekly_seasonality=False, 
        # daily_seasonality=False,
        changepoint_prior_scale=changepoint_prior_scale,
        seasonality_mode=seasonality_mode,
        seasonality_prior_scale=seasonality_prior_scale
    )
    model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
    
    # Melatih model dengan data train
    model.fit(train_data)

    # Membuat dataframe masa depan yang mencakup periode data test
    future = model.make_future_dataframe(periods=len(test_data), freq='MS')

    # Melakukan prediksi
    forecast = model.predict(future)

    # Mengambil prediksi yang sesuai dengan data test
    forecast_test = forecast[-len(test_data):]

    # Menggabungkan data asli dengan prediksi
    test_data.loc[:, 'yhat'] = forecast_test['yhat'].values

    # Menghitung SMAPE untuk data test
    smape_value = smape(test_data['y'], test_data['yhat'])
    print(f"SMAPE dengan changepoint_prior_scale={changepoint_prior_scale}, seasonality_mode={seasonality_mode}, seasonality_prior_scale={seasonality_prior_scale}: {smape_value:.2f}%")

    # Menyimpan kombinasi parameter terbaik
    if smape_value < best_smape:
        best_smape = smape_value
        best_params = (changepoint_prior_scale, seasonality_mode, seasonality_prior_scale)

print(f"\nParameter terbaik: changepoint_prior_scale={best_params[0]}, seasonality_mode={best_params[1]}, seasonality_prior_scale={best_params[2]}")
print(f"SMAPE terbaik: {best_smape:.2f}%")

# Latih model dengan seluruh data menggunakan parameter terbaik
final_model = Prophet(
    yearly_seasonality=True, 
    # weekly_seasonality=False, 
    # daily_seasonality=False,
    changepoint_prior_scale=best_params[0],
    seasonality_mode=best_params[1],
    seasonality_prior_scale=best_params[2]
)
final_model.add_seasonality(name='monthly', period=30.5, fourier_order=5)

# Melatih model dengan seluruh data
final_model.fit(df_Revenue)

# Membuat dataframe masa depan untuk 6 bulan ke depan
future = final_model.make_future_dataframe(periods=12, freq='MS')

# Melakukan prediksi
forecast_final = final_model.predict(future)

# Menampilkan hasil prediksi
print(forecast_final[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])

# Menyimpan hasil prediksi ke dalam DataFrame
forecast_df_Revenue = forecast_final[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
forecast_df_Revenue.columns = ['Date', 'Forecast', 'Lower Bound', 'Upper Bound']

forecast_df_Revenue['Forecast'] = forecast_df_Revenue['Forecast'].round(2)
forecast_df_Revenue['Lower Bound'] = forecast_df_Revenue['Lower Bound'].round(2)
forecast_df_Revenue['Upper Bound'] = forecast_df_Revenue['Upper Bound'].round(2)

forecast_df_Revenue.loc[:, 'Date'] = pd.to_datetime(forecast_df_Revenue['Date'])
# Menambahkan data untuk 'Forecast'
forecast_rows = pd.DataFrame({
    'ds': forecast_df_Revenue['Date'],
    'y': forecast_df_Revenue['Forecast'],
    'jenis': 'forecast',
    'Breakdown' : 'Total Revenue'
})

# Menambahkan data untuk 'Lower Bound'
lower_bound_rows = pd.DataFrame({
    'ds': forecast_df_Revenue['Date'],
    'y': forecast_df_Revenue['Lower Bound'],
    'jenis': 'lower bound',
    'Breakdown' : 'Total Revenue'
})

# Menambahkan data untuk 'Upper Bound'
upper_bound_rows = pd.DataFrame({
    'ds': forecast_df_Revenue['Date'],
    'y': forecast_df_Revenue['Upper Bound'],
    'jenis': 'upper bound',
    'Breakdown' : 'Total Revenue'
})
df_Revenue_actual = pd.DataFrame({
    'ds': df_Revenue['ds'],
    'y': df_Revenue['y'],
    'jenis': 'actual',
    'Breakdown' : 'Total Revenue'
})
# Menggabungkan semua DataFrame
result_df_Revenue = pd.concat([df_Revenue_actual,forecast_rows, lower_bound_rows, upper_bound_rows], ignore_index=True)

# Cash & Deposit
print("--"*60)
print("CASH AND DEPOSIT")

df_Cash_Deposit = df_overview[df_overview['BreakDown2']=='Cash and Deposit'][['Date','amount']]
# Reset index
df_Cash_Deposit = df_Cash_Deposit.reset_index(drop=True)
df_Cash_Deposit.columns = ['ds','y']
df_Cash_Deposit['ds'] = pd.to_datetime(df_Cash_Deposit['ds'])

#  Kombinasi parameter untuk tuning SARIMA
p = d = q = range(0, 2)  # AR, I, MA order
P = D = Q = range(0, 2)  # Seasonal ARIMA order
s = [12]  # Musiman dalam bulan (12 bulan)

# Membuat list kombinasi parameter p, d, q, P, D, Q, s
param_combinations = list(itertools.product(p, d, q, P, D, Q, s))

# Menyimpan hasil tuning
best_smape = float('inf')
best_params = None

# Train dan test split
train_size = int(0.8 * len(df_Cash_Deposit))
train_data = df_Cash_Deposit[:train_size]
test_data = df_Cash_Deposit[train_size:]

# Fungsi SMAPE
def smape(actual, predicted):
    return 100 * np.mean(2 * np.abs(predicted - actual) / (np.abs(predicted) + np.abs(actual)))

# Looping untuk setiap kombinasi parameter
for (p, d, q, P, D, Q, s) in param_combinations:
    try:
        # Inisialisasi model SARIMA dengan parameter tuning
        model = SARIMAX(train_data['y'],
                        order=(p, d, q),
                        seasonal_order=(P, D, Q, s),
                        enforce_stationarity=False,
                        enforce_invertibility=False)

        # Melatih model dengan data train
        model_fit = model.fit(disp=False)

        # Melakukan prediksi
        forecast = model_fit.forecast(steps=len(test_data))

        # Menggabungkan data asli dengan prediksi
        test_data.loc[:, 'yhat'] = forecast

        # Menghitung SMAPE untuk data test
        smape_value = smape(test_data['y'], test_data['yhat'])
        print(f"SMAPE dengan order=({p},{d},{q}), seasonal_order=({P},{D},{Q},{s}): {smape_value:.2f}%")

        # Menyimpan kombinasi parameter terbaik
        if smape_value < best_smape:
            best_smape = smape_value
            best_params = (p, d, q, P, D, Q, s)

    except Exception as e:
        print(f"Error untuk parameter {p, d, q, P, D, Q, s}: {e}")

print(f"\nParameter terbaik: order=({best_params[0]},{best_params[1]},{best_params[2]}), seasonal_order=({best_params[3]},{best_params[4]},{best_params[5]},{best_params[6]})")
print(f"SMAPE terbaik: {best_smape:.2f}%")

# Membuat model SARIMAX menggunakan parameter terbaik
final_model = SARIMAX(df_Cash_Deposit['y'],
                     order=(best_params[0], best_params[1], best_params[2]),
                     seasonal_order=(best_params[3], best_params[4], best_params[5], best_params[6]),
                     enforce_stationarity=False,
                     enforce_invertibility=False)

# Melatih model dengan seluruh data
final_model_fit = final_model.fit(disp=False)

# Prediksi untuk seluruh data historis (dari awal data hingga akhir)
forecast_historic = final_model_fit.get_prediction(steps=len(df_Cash_Deposit))
forecast_historic_mean = forecast_historic.predicted_mean
forecast_historic_ci = forecast_historic.conf_int()

# Prediksi untuk 12 bulan ke depan
forecast_future = final_model_fit.get_forecast(steps=12)
forecast_future_mean = forecast_future.predicted_mean
forecast_future_ci = forecast_future.conf_int()

# Membuat tanggal untuk prediksi masa depan
future_dates = pd.date_range(start=df_Cash_Deposit['ds'].max() + pd.DateOffset(months=1), periods=12, freq='MS')

# Konversi 'future_dates' ke Series agar bisa digabungkan dengan tanggal historis
future_dates_series = pd.Series(future_dates)

# Menyimpan hasil prediksi ke dalam DataFrame untuk data historis dan data masa depan
forecast_df_Cash_Deposit = pd.DataFrame({
    'Date': pd.concat([df_Cash_Deposit['ds'].reset_index(drop=True), future_dates_series], ignore_index=True),
    'Forecast': pd.concat([forecast_historic_mean, forecast_future_mean], ignore_index=True),
    'Lower Bound': pd.concat([forecast_historic_ci.iloc[:, 0], forecast_future_ci.iloc[:, 0]], ignore_index=True),
    'Upper Bound': pd.concat([forecast_historic_ci.iloc[:, 1], forecast_future_ci.iloc[:, 1]], ignore_index=True)
})

forecast_df_Cash_Deposit['Forecast'] = forecast_df_Cash_Deposit['Forecast'].round(2)
forecast_df_Cash_Deposit['Lower Bound'] = forecast_df_Cash_Deposit['Lower Bound'].round(2)
forecast_df_Cash_Deposit['Upper Bound'] = forecast_df_Cash_Deposit['Upper Bound'].round(2)

forecast_df_Cash_Deposit.loc[:, 'Date'] = pd.to_datetime(forecast_df_Cash_Deposit['Date'])
# Menambahkan data untuk 'Forecast'
forecast_rows = pd.DataFrame({
    'ds': forecast_df_Cash_Deposit['Date'],
    'y': forecast_df_Cash_Deposit['Forecast'],
    'jenis': 'forecast',
    'Breakdown' : 'Cash and Deposit'
})

# Menambahkan data untuk 'Lower Bound'
lower_bound_rows = pd.DataFrame({
    'ds': forecast_df_Cash_Deposit['Date'],
    'y': forecast_df_Cash_Deposit['Lower Bound'],
    'jenis': 'lower bound',
    'Breakdown' : 'Cash and Deposit'
})

# Menambahkan data untuk 'Upper Bound'
upper_bound_rows = pd.DataFrame({
    'ds': forecast_df_Cash_Deposit['Date'],
    'y': forecast_df_Cash_Deposit['Upper Bound'],
    'jenis': 'upper bound',
    'Breakdown' : 'Cash and Deposit'
})
df_Cash_Deposit_actual = pd.DataFrame({
    'ds': df_Cash_Deposit['ds'],
    'y': df_Cash_Deposit['y'],
    'jenis': 'actual',
    'Breakdown' : 'Cash and Deposit'
})
# Menggabungkan semua DataFrame
result_df_Cash_Deposit = pd.concat([df_Cash_Deposit_actual,forecast_rows, lower_bound_rows, upper_bound_rows], ignore_index=True)

# EBITDA
print("--"*60)
print("EBITDA")

df_EBITDA = df_overview[df_overview['BreakDown2']=='EBITDA After Minority'][['Date','amount']]
# Reset index
df_EBITDA = df_EBITDA.reset_index(drop=True)
df_EBITDA.columns = ['ds','y']
df_EBITDA['ds'] = pd.to_datetime(df_EBITDA['ds'])

# Kombinasi parameter untuk tuning
changepoint_prior_scales = [ 0.1, 0.2]
seasonality_modes = ['additive','multiplicative']
seasonality_prior_scales = [0.1, 0.2]

# Membuat list kombinasi parameter
param_combinations = list(itertools.product(changepoint_prior_scales, seasonality_modes, seasonality_prior_scales))

# Fungsi untuk menghitung SMAPE
def smape(actual, predicted):
    return 100 * np.mean(2 * np.abs(predicted - actual) / (np.abs(predicted) + np.abs(actual)))

# Menyimpan hasil tuning
best_smape = float('inf')
best_params = None

# Train dan test split
train_size = int(0.8 * len(df_EBITDA))
train_data = df_EBITDA[:train_size]
test_data = df_EBITDA[train_size:]

# Looping untuk setiap kombinasi parameter
for changepoint_prior_scale, seasonality_mode, seasonality_prior_scale in param_combinations:
    
    # Inisialisasi model Prophet dengan parameter tuning
    model = Prophet(
        yearly_seasonality=True, 
        # weekly_seasonality=False, 
        # daily_seasonality=False,
        changepoint_prior_scale=changepoint_prior_scale,
        seasonality_mode=seasonality_mode,
        seasonality_prior_scale=seasonality_prior_scale
    )
    model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
    
    # Melatih model dengan data train
    model.fit(train_data)

    # Membuat dataframe masa depan yang mencakup periode data test
    future = model.make_future_dataframe(periods=len(test_data), freq='MS')

    # Melakukan prediksi
    forecast = model.predict(future)

    # Mengambil prediksi yang sesuai dengan data test
    forecast_test = forecast[-len(test_data):]

    # Menggabungkan data asli dengan prediksi
    test_data.loc[:, 'yhat'] = forecast_test['yhat'].values

    # Menghitung SMAPE untuk data test
    smape_value = smape(test_data['y'], test_data['yhat'])
    print(f"SMAPE dengan changepoint_prior_scale={changepoint_prior_scale}, seasonality_mode={seasonality_mode}, seasonality_prior_scale={seasonality_prior_scale}: {smape_value:.2f}%")

    # Menyimpan kombinasi parameter terbaik
    if smape_value < best_smape:
        best_smape = smape_value
        best_params = (changepoint_prior_scale, seasonality_mode, seasonality_prior_scale)

print(f"\nParameter terbaik: changepoint_prior_scale={best_params[0]}, seasonality_mode={best_params[1]}, seasonality_prior_scale={best_params[2]}")
print(f"SMAPE terbaik: {best_smape:.2f}%")

# Latih model dengan seluruh data menggunakan parameter terbaik
final_model = Prophet(
    yearly_seasonality=True, 
    # weekly_seasonality=False, 
    # daily_seasonality=False,
    changepoint_prior_scale=best_params[0],
    seasonality_mode=best_params[1],
    seasonality_prior_scale=best_params[2]
)
final_model.add_seasonality(name='monthly', period=30.5, fourier_order=5)

# Melatih model dengan seluruh data
final_model.fit(df_EBITDA)

# Membuat dataframe masa depan untuk 6 bulan ke depan
future = final_model.make_future_dataframe(periods=12, freq='MS')

# Melakukan prediksi
forecast_final = final_model.predict(future)

# Menampilkan hasil prediksi
print(forecast_final[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])

# Menyimpan hasil prediksi ke dalam DataFrame
forecast_df_EBITDA = forecast_final[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
forecast_df_EBITDA.columns = ['Date', 'Forecast', 'Lower Bound', 'Upper Bound']

forecast_df_EBITDA['Forecast'] = forecast_df_EBITDA['Forecast'].round(2)
forecast_df_EBITDA['Lower Bound'] = forecast_df_EBITDA['Lower Bound'].round(2)
forecast_df_EBITDA['Upper Bound'] = forecast_df_EBITDA['Upper Bound'].round(2)

forecast_df_EBITDA.loc[:, 'Date'] = pd.to_datetime(forecast_df_EBITDA['Date'])
# Menambahkan data untuk 'Forecast'
forecast_rows = pd.DataFrame({
    'ds': forecast_df_EBITDA['Date'],
    'y': forecast_df_EBITDA['Forecast'],
    'jenis': 'forecast',
    'Breakdown' : 'EBITDA After Minority'
})

# Menambahkan data untuk 'Lower Bound'
lower_bound_rows = pd.DataFrame({
    'ds': forecast_df_EBITDA['Date'],
    'y': forecast_df_EBITDA['Lower Bound'],
    'jenis': 'lower bound',
    'Breakdown' : 'EBITDA After Minority'
})

# Menambahkan data untuk 'Upper Bound'
upper_bound_rows = pd.DataFrame({
    'ds': forecast_df_EBITDA['Date'],
    'y': forecast_df_EBITDA['Upper Bound'],
    'jenis': 'upper bound',
    'Breakdown' : 'EBITDA After Minority'
})
df_EBITDA_actual = pd.DataFrame({
    'ds': df_EBITDA['ds'],
    'y': df_EBITDA['y'],
    'jenis': 'actual',
    'Breakdown' : 'EBITDA After Minority'
})
# Menggabungkan semua DataFrame
result_df_EBITDA = pd.concat([df_EBITDA_actual,forecast_rows, lower_bound_rows, upper_bound_rows], ignore_index=True)

# Net Profit
print("--"*60)
print("NET PROFIT")

df_Netprofit = df_overview[df_overview['BreakDown2']=='Net Profit After Minority'][['Date','amount']]
# Reset index
df_Netprofit = df_Netprofit.reset_index(drop=True)
df_Netprofit.columns = ['ds','y']
df_Netprofit['ds'] = pd.to_datetime(df_Netprofit['ds'])

# Kombinasi parameter untuk tuning
changepoint_prior_scales = [ 0.1,0.2, 0.5]
seasonality_modes = ['multiplicative']
seasonality_prior_scales = [ 0.1,0.2]

# Membuat list kombinasi parameter
param_combinations = list(itertools.product(changepoint_prior_scales, seasonality_modes, seasonality_prior_scales))

# Fungsi untuk menghitung SMAPE
def smape(actual, predicted):
    return 100 * np.mean(2 * np.abs(predicted - actual) / (np.abs(predicted) + np.abs(actual)))

# Menyimpan hasil tuning
best_smape = float('inf')
best_params = None

# Train dan test split
train_size = int(0.8 * len(df_Netprofit))
train_data = df_Netprofit[:train_size]
test_data = df_Netprofit[train_size:]

# Looping untuk setiap kombinasi parameter
for changepoint_prior_scale, seasonality_mode, seasonality_prior_scale in param_combinations:
    
    # Inisialisasi model Prophet dengan parameter tuning
    model = Prophet(
        yearly_seasonality=True, 
        # weekly_seasonality=False, 
        # daily_seasonality=False,
        changepoint_prior_scale=changepoint_prior_scale,
        seasonality_mode=seasonality_mode,
        seasonality_prior_scale=seasonality_prior_scale
    )
    model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
    
    # Melatih model dengan data train
    model.fit(train_data)

    # Membuat dataframe masa depan yang mencakup periode data test
    future = model.make_future_dataframe(periods=len(test_data), freq='MS')

    # Melakukan prediksi
    forecast = model.predict(future)

    # Mengambil prediksi yang sesuai dengan data test
    forecast_test = forecast[-len(test_data):]

    # Menggabungkan data asli dengan prediksi
    test_data.loc[:, 'yhat'] = forecast_test['yhat'].values

    # Menghitung SMAPE untuk data test
    smape_value = smape(test_data['y'], test_data['yhat'])
    print(f"SMAPE dengan changepoint_prior_scale={changepoint_prior_scale}, seasonality_mode={seasonality_mode}, seasonality_prior_scale={seasonality_prior_scale}: {smape_value:.2f}%")

    # Menyimpan kombinasi parameter terbaik
    if smape_value < best_smape:
        best_smape = smape_value
        best_params = (changepoint_prior_scale, seasonality_mode, seasonality_prior_scale)

print(f"\nParameter terbaik: changepoint_prior_scale={best_params[0]}, seasonality_mode={best_params[1]}, seasonality_prior_scale={best_params[2]}")
print(f"SMAPE terbaik: {best_smape:.2f}%")

# Latih model dengan seluruh data menggunakan parameter terbaik
final_model = Prophet(
    yearly_seasonality=True, 
    # weekly_seasonality=False, 
    # daily_seasonality=False,
    changepoint_prior_scale=best_params[0],
    seasonality_mode=best_params[1],
    seasonality_prior_scale=best_params[2]
)
final_model.add_seasonality(name='monthly', period=30.5, fourier_order=5)

# Melatih model dengan seluruh data
final_model.fit(df_Netprofit)

# Membuat dataframe masa depan untuk 6 bulan ke depan
future = final_model.make_future_dataframe(periods=12, freq='MS')

# Melakukan prediksi
forecast_final = final_model.predict(future)

# Menampilkan hasil prediksi
print(forecast_final[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])

# Menyimpan hasil prediksi ke dalam DataFrame
forecast_df_Netprofit = forecast_final[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
forecast_df_Netprofit.columns = ['Date', 'Forecast', 'Lower Bound', 'Upper Bound']

forecast_df_Netprofit['Forecast'] = forecast_df_Netprofit['Forecast'].round(2)
forecast_df_Netprofit['Lower Bound'] = forecast_df_Netprofit['Lower Bound'].round(2)
forecast_df_Netprofit['Upper Bound'] = forecast_df_Netprofit['Upper Bound'].round(2)

forecast_df_Netprofit.loc[:, 'Date'] = pd.to_datetime(forecast_df_Netprofit['Date'])
# Menambahkan data untuk 'Forecast'
forecast_rows = pd.DataFrame({
    'ds': forecast_df_Netprofit['Date'],
    'y': forecast_df_Netprofit['Forecast'],
    'jenis': 'forecast',
    'Breakdown' : 'Net Profit After Minority'
})

# Menambahkan data untuk 'Lower Bound'
lower_bound_rows = pd.DataFrame({
    'ds': forecast_df_Netprofit['Date'],
    'y': forecast_df_Netprofit['Lower Bound'],
    'jenis': 'lower bound',
    'Breakdown' : 'Net Profit After Minority'
})

# Menambahkan data untuk 'Upper Bound'
upper_bound_rows = pd.DataFrame({
    'ds': forecast_df_Netprofit['Date'],
    'y': forecast_df_Netprofit['Upper Bound'],
    'jenis': 'upper bound',
    'Breakdown' : 'Net Profit After Minority'
})
df_Netprofit_actual = pd.DataFrame({
    'ds': df_Netprofit['ds'],
    'y': df_Netprofit['y'],
    'jenis': 'actual',
    'Breakdown' : 'Net Profit After Minority'
})
# Menggabungkan semua DataFrame
result_df_Netprofit = pd.concat([df_Netprofit_actual,forecast_rows, lower_bound_rows, upper_bound_rows], ignore_index=True)


# print("--"*60)
# print("DF FORECAST")

df_Forecast_all = pd.concat([result_df_production,result_df_Transport,result_df_SR, result_df_Sales, result_df_coal_cast_cost, result_df_Avg_SP, result_df_Revenue, result_df_Cash_Deposit,result_df_EBITDA, result_df_Netprofit],ignore_index=True)
# df_Forecast_all = pd.concat([result_df_production,result_df_Transport,result_df_SR, result_df_Sales, result_df_coal_cast_cost, result_df_Avg_SP],ignore_index=True)
df_Forecast_all.rename(columns={'ds': 'Date', 'y': 'Amount', 'jenis':'Tipe','Breakdown':'BreakDown'}, inplace=True)
# df_Forecast_all

from datetime import datetime
df_Forecast_all['last_updated'] = datetime.now()
df_Forecast_all['Updated_Date'] = datetime.now()
# df_Forecast_all


# print("--"*60)
# print("DF LAMA")

df_overview_filter.copy()
df_overview_non_actual = df_overview_filter[df_overview_filter['tipe'] != "Actual"]
df_overview_non_actual['Date'] = pd.to_datetime(df_overview_non_actual['Date'])
# DF non actual
df_overview_non_Actual_group = df_overview_non_actual.groupby(['Date','BreakDown2','tipe','last_updated','Updated_Date'])['amount'].sum().reset_index()
df_overview_non_Actual_group.rename(columns={'amount': 'Amount', 'tipe':'Tipe','BreakDown2':'BreakDown'}, inplace=True)

print("--"*60)
print("INSERT DATA TO DATABASE")


# gabungkan semua
df_Final = pd.concat([df_Forecast_all, df_overview_non_Actual_group],ignore_index=True)
df_Final.rename(columns={'Amount': 'amount', 'Tipe':'tipe'}, inplace=True)
df_Final['amount'] = df_Final['amount'].round(2)  # Membatasi hingga 2 angka desimal
ordered_columns = ['Date', 'BreakDown', 'tipe', 'amount', 'last_updated', 'Updated_Date']
df_Final['amount'] = df_Final['amount'].astype(float)
df_Final = df_Final.reindex(columns=ordered_columns)
# df_Final

df_Final[['last_updated', 'Updated_Date']] = df_Final[['last_updated', 'Updated_Date']].apply(pd.to_datetime)


import pyodbc
from dotenv import load_dotenv
import os
import pandas as pd

# Muat variabel lingkungan dari file .env
# load_dotenv()
server = '10.3.4.139,1433'
database = 'dwh_prod'
username = 'developerptba'
password = 'LnfPYVFW1K2TAKf'
# Hardcode nilai koneksi SQL Server
# server = os.environ['SERVER']
# database = os.environ['DATABASE']
# username = os.environ['USERNAME_1']
# password = os.environ['PASSWORD']

# Koneksi ke SQL Server
try:
    conn = pyodbc.connect(f'DRIVER={{SQL Server}};SERVER={server};DATABASE={database};UID={username};PWD={password}')
    # conn = pyodbc.connect(f'DRIVER={{SQL Server}};SERVER={server};DATABASE={database};UID={username};PWD={password}')
    print("Connection successful")
except Exception as e:
    print(f"Connection failed: {e}")
    exit(1)  # Keluar jika koneksi gagal

# Melakukan operasi lainnya seperti truncate tabel dan insert data
cursor = conn.cursor()

# Truncate tabel
truncate_query = "TRUNCATE TABLE dwh.DM_forecast_financial_position"
try:
    cursor.execute(truncate_query)
    print("Table truncated successfully.")
except pyodbc.Error as e:
    print(f"Error truncating table: {e}")
    conn.rollback()

# Setting fast_executemany
cursor.fast_executemany = True

insert_query = """
INSERT INTO dwh.DM_forecast_financial_position
([Date], [BreakDown], [tipe], [amount], [last_updated], [Updated_Date]) 
VALUES (?, ?, ?, ?, ?, ?)
"""

# Siapkan data yang akan di-insert
rows_to_insert = [row for row in df_Final.itertuples(index=False)]
print(f"{len(rows_to_insert)} rows will be inserted into the database.")

# Nonaktifkan autocommit
conn.autocommit = False

# Menggunakan try-except saat insert
try:
    for row in rows_to_insert:
        cursor.execute(insert_query, row)
    # Commit perubahan jika semua insert berhasil
    conn.commit()
    print("Data successfully inserted.")
except pyodbc.Error as e:
    print(f"Error inserting rows: {e}")
    conn.rollback()  # Rollback jika terjadi error

# Menutup koneksi
conn.close()
print("Forecasting Done")

