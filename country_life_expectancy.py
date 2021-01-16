import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.compose import ColumnTransformer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# read csv and load into dataframe
dataset = pd.read_csv('life_expectancy.csv')

# check the first entries of data
print(dataset.head())
print(dataset.describe())

# drop a column in order to generalize overpip
dataset = dataset.drop(['Country'], axis=1)

# to split the labels and features
# create last column
labels = dataset.iloc[:, -1]

# assign subset of columns from first to second last
features = dataset.iloc[:, 0:-1]
# convert categorical columns into numerical col using one-hot-encoding
features = pd.get_dummies(features)

# split data into training and test sets
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.20,
                                                                            random_state=23)

# normalize the numerical features
numerical_features = features.select_dtypes(include=['float64', 'int64'])
numerical_columns = numerical_features.columns

ct = ColumnTransformer([("only numeric", StandardScaler(), numerical_columns)], remainder='passthrough')

# fit the training data and transform
features_train_scaled = ct.fit_transform(features_train)
features_test_scaled = ct.fit_transform(features_test)

# building the model
my_model = Sequential()
# creating input layer with shape corresponding to the number of input features
input = InputLayer(input_shape=(features.shape[1],))
# add the input layer to the model
my_model.add(input)
# add hidden layer with one neuron
my_model.add(Dense(1, activation='relu'))
# add output layer with one neuron
my_model.add(Dense(1))
print(my_model.summary())
# optimize the parameters/model with learning rate
opt = Adam(learning_rate=0.01)
# compile the model for loss estimation
my_model.compile(loss='mse', metrics=['mae'], optimizer=opt)

# fit and evaluate the model
my_model.fit(features_train_scaled, labels_train, epochs=40, batch_size=1, verbose=1)
# evaluate the trained model
res_mse, res_mae = my_model.evaluate(features_test_scaled, labels_test, verbose=0)

# print final loss 'root mean square error' and final metric 'mean absolute error'

print('res_mse:', res_mse)
print('res_mae:', res_mae)
