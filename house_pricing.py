import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the dataset
df = pd.read_csv('CAhousing.csv', sep=',')

# Separate numerical and categorical columns
num_cols = df.select_dtypes(include=['int64', 'float64']).columns
cat_cols = df.select_dtypes(include=['object']).columns

# Check for missing values in numerical columns and fill them
df[num_cols] = df[num_cols].fillna(df[num_cols].mean())

# Check for missing values in categorical columns and fill them
for col in cat_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)

# Visualize the distribution of target variable
sns.set()
plt.figure(figsize=(10,6))
sns.histplot(df['median_house_value'], bins=30, kde=True)
plt.title('Distribution of median_house_value')
plt.show()

sns.set()
plt.figure(figsize=(10,6))
sns.scatterplot(x = df['median_house_value'], y =df['median_income'])
plt.title('Median Income vs Median House Value')
plt.show()

# Apply one-hot encoding to 'ocean_proximity'
df = pd.get_dummies(df, columns=['ocean_proximity'], prefix="", prefix_sep="")

# Separate target variable
target = df['median_house_value']
df_features = df.drop('median_house_value', axis=1)

# Feature scaling: standardization
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df_features), columns=df_features.columns)

# Append target back to the DataFrame
df_scaled = pd.concat([df_scaled, target.reset_index(drop=True)], axis=1)

# Splitting the data
X = df_scaled.drop('median_house_value', axis=1)
y = df_scaled['median_house_value']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions and calculate mean squared error
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)
mse_train = mean_squared_error(y_train, y_train_pred)
mse_test = mean_squared_error(y_test, y_test_pred)

print('Training MSE:', mse_train)
print('Test MSE:', mse_test)

# Predicted vs Actual scatter plot
plt.scatter(y_test, y_test_pred)
plt.xlabel('Actual values')
plt.ylabel('Predicted values')
plt.title('Actual vs Predicted values')

# Line of best fit
m, b = np.polyfit(y_test, y_test_pred, 1)
plt.plot(y_test, m*y_test + b, color='red')

plt.show()

# Retrieve the 'median_income' column from the test data.
# Remember, since 'median_income' is standardized, we need to inverse transform it back to original scale.
median_income_test = scaler.inverse_transform(X_test)[:, df_features.columns.get_loc('median_income')]

# Create scatter plots
plt.scatter(median_income_test, y_test, color='blue', label='Actual values')
plt.scatter(median_income_test, y_test_pred, color='red', label='Predicted values')
plt.xlabel('Median Income')
plt.ylabel('House Value')
plt.title('Median Income vs House Value')
plt.legend()

plt.show()

# Assuming this is your new data (make sure it has the same structure as your training data)
new_data = {
    'longitude': [-122.23],
    'latitude': [37.88],
    'housing_median_age': [41.0],
    'total_rooms': [880.0],
    'total_bedrooms': [129.0],
    'population': [322.0],
    'households': [126.0],
    'median_income': [8.3252],
    'ocean_proximity': ['NEAR BAY']
}

# Convert the dictionary into a DataFrame
new_data_df = pd.DataFrame(new_data)

# One-hot encoding for 'ocean_proximity'
new_data_df = pd.get_dummies(new_data_df, columns=['ocean_proximity'], prefix="", prefix_sep="")

# Ensure new data has all the columns that the model was trained on
missing_cols = set(df_features.columns) - set(new_data_df.columns)
for c in missing_cols:
    new_data_df[c] = 0

# Ensure the order of column in the test query is in the same order than in train set
new_data_df = new_data_df[df_features.columns]

# Ensure all column names are strings
new_data_df.columns = new_data_df.columns.astype(str)

# Standardizing the features
new_features_scaled = scaler.transform(new_data_df)

# Convert scaled features to DataFrame and ensure it has correct column names
new_features_df = pd.DataFrame(new_features_scaled, columns=df_features.columns)

# Now you can use the trained model to make predictions on your new data
new_prediction = model.predict(new_features_df)

print("Predicted house value: ", new_prediction)


