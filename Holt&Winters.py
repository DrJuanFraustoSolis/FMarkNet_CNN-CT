import pandas as pd
import statsmodels.api as sm

# Cargar tu serie temporal en un DataFrame
# Asumiendo que tienes una columna llamada "valor" con los datos
data = pd.read_csv('tu_serie_temporal.csv')

# Ajustar el modelo de Holt-Winters
model = sm.tsa.ExponentialSmoothing(data['valor'], trend='add', seasonal='add', seasonal_periods=m)
fit = model.fit()

# Realizar pronósticos para futuros períodos
n = 12  # Número de períodos a pronosticar
forecast = fit.forecast(steps=n)

# Los pronósticos estarán en la variable "forecast"
print(forecast)
