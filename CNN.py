# univariate multi-step vector-output 1d cnn example
import numpy as np
from numpy import array
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Serie1.csv')
Dato = dataset.iloc[:, 1:2].values
Dato = np.array(Dato)
Dato = Dato.T
# print(Dato[0])
N_Meses = 1
Componentes = [0] * int(len(Dato[0]) / N_Meses)

C = [Dato[0][i:i + N_Meses] for i in range(0, len(Dato[0]), N_Meses)]

for i in range(len(Componentes)):
    Componentes[i] = sum(C[i]) / N_Meses
    #print(Componentes[i])


# split a univariate sequence into samples
def split_sequence(sequence, n_steps_in, n_steps_out):
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out
        # check if we are beyond the sequence
        if out_end_ix > len(sequence):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)


# define input sequence
raw_seq = Dato[0]
# choose a number of time steps
n_steps_in, n_steps_out = 24, 240
# split into samples
X, y = split_sequence(raw_seq, n_steps_in, n_steps_out)
# reshape from [samples, timesteps] into [samples, timesteps, features]
n_features = 1
X = X.reshape((X.shape[0], X.shape[1], n_features))
# define model
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(n_steps_in, n_features)))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(50, activation='relu'))
model.add(Dense(n_steps_out))
model.compile(optimizer='adam', loss='mse')
# fit model
model.fit(X, y, epochs=2000, verbose=0)
# demonstrate prediction
l = [0] * n_steps_in
n = n_steps_in
for i in range(n_steps_in):
    l[i] = Componentes[(len(Componentes)) - n]
    n = n - 1
#print(l)
x_input = np.array(l)
x_input = x_input.reshape((1, n_steps_in, n_features))
yhat = np.array(model.predict(x_input, verbose=0).T)
print(yhat)
''''
if(N_Meses == 1):
    yreal = [15.6, 18.26, 19.59, 22.45, 23.8, 26.02, 25.56, 25.27]
elif(N_Meses == 3):
    yreal = [17.81666667, 24.09, 24.89666667, 18.70666667, 17.13666667, 23.69, 25.59666667, 19.66]
elif(N_Meses == 4):
    yreal = [18.975, 25.1625, 19.995, 18.4025, 25.1475, 21.0125, 21.8525, 22.5275]
elif (N_Meses == 6):
    yreal = [20.95, 21.80166667, 20.41333333, 22.62833333, 22.06, 22.36666667, 18.225, 18.21166667]

print("prediccion: ", yhat.T)
print("Real: ", yreal)
mape = np.mean(np.abs((yreal - yhat) / yreal)) * 100
print("MAPE:", mape)
x=np.arange(0,8,1)
plt.plot(x,np.array(yreal),'blue')
plt.plot(x,np.array(yhat),'red')
plt.show()
'''''