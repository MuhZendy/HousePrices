# The three libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# sklean libraries
import sklearn
from sklearn.metrics import mean_absolute_error
from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# keras
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping

# Setup
X_full = pd.read_csv("data/train.csv", index_col='Id')
X_full.dropna(axis=0, subset=["SalePrice"], inplace=True)
y_full = X_full.pop("SalePrice")
y_full = np.log1p(y_full)

# Get only numeric and categorical columns
num_col = [cname for cname in X_full.columns if X_full[cname].dtype in ['int64', 'float64']]
cat_col = [cname for cname in X_full.columns if X_full[cname].dtype == "str"]

# num_col = ['LotArea', '1stFlrSF', '2ndFlrSF', 'YearBuilt', 'TotRmsAbvGrd', 'PoolArea', 'GarageArea', 'MiscVal', 'TotalBsmtSF']
# cat_col = ['OverallQual', 'OverallCond', 'GarageQual', 'GarageCond', 'PoolQC', 'BsmtQual', 'BsmtCond', 'HeatingQC', 'CentralAir', 'Heating']

X_full = X_full[num_col + cat_col].copy()

# Split X_full into X_train, y_train, X_valid, y_valid
X_train, X_valid, y_train, y_valid = train_test_split(X_full, y_full, train_size=0.8, \
                                                      test_size=0.2, random_state=0)

# Make strategies to handle numeric and categorical columns
num_transformer = SimpleImputer(strategy='median')
cat_transformer = make_pipeline(
    SimpleImputer(strategy='most_frequent'),
    OneHotEncoder(handle_unknown='ignore')
)

# Final preprocessing strategy
preprocessor = make_column_transformer(
    (num_transformer, num_col),
    (cat_transformer, cat_col)
)

# Preprocess
X_train = preprocessor.fit_transform(X_train)
X_valid = preprocessor.transform(X_valid)
print(X_train.shape)

# Deep learning model
model = keras.Sequential([
    layers.BatchNormalization(input_shape=[X_train.shape[1]]),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.1),
    layers.BatchNormalization(),
    layers.Dense(256, activation='selu'),
    layers.Dropout(0.1),
    layers.BatchNormalization(),
    layers.Dense(256, activation='relu'),
    layers.Dense(1)
])

model.compile(optimizer='adam', loss='mae')
early_stopping = EarlyStopping(min_delta=10, patience=20, \
                               restore_best_weights=True)

history = model.fit(X_train, y_train, \
                    validation_data=(X_valid, y_valid), \
                    batch_size=128, epochs=1000, \
                    callbacks=[early_stopping])

history_df = pd.DataFrame(history.history)
history_df.loc[:, ['loss', 'val_loss']].plot()
plt.show()

