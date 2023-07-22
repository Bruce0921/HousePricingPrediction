import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# df = pd.read_csv('CAhousing.csv', delim_whitespace=True)
df = pd.read_csv('CAhousing.csv', sep=',')
print(df.columns)
# df.head() will display the first few rows of the dataset.
# df.info() will give a summary of the dataset, including the number of non-null entries in each column.
# df.describe() will provide descriptive statistics for each column.

# print(df.head())
# print(df.info())
# print(df.describe())

# print(df.isnull().sum())

# Set the style for seaborn
sns.set()

# Create the histogram
plt.figure(figsize=(10,6))
sns.histplot(df['median_house_value'], bins=30, kde=True)
plt.title('Distribution of median_house_value')
plt.show()

# data preprocessing
# Feature scaling:standardization or normalization
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
# Encoding Categorical Variables
# df_encoded = pd.get_dummies(df, columns=['ocean_proximity'])
# Separate target variable
target = df['median_house_value']
df_features = df.drop('median_house_value', axis=1)

# Apply one-hot encoding to 'ocean_proximity'
encoder = OneHotEncoder()
ocean_proximity_encoded = encoder.fit_transform(df_features[['ocean_proximity']]).toarray()

# Construct new dataframe with encoded 'ocean_proximity' and drop the original column
df_encoded = pd.DataFrame(ocean_proximity_encoded, columns=encoder.categories_[0])
df_features = pd.concat([df_features.drop('ocean_proximity', axis=1), df_encoded], axis=1)

# Scale numeric features only
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df_features), columns=df_features.columns)

# Append target back to the DataFrame
df_scaled = pd.concat([df_scaled, target], axis=1)


# Splitting the Data, This allows you to evaluate how well your model will perform on unseen data
X = df.drop('median_house_value', axis=1)
y = df['median_house_value']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
