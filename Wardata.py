import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# load the dataset
file_path = r'C:\Users\lokey maddy\Downloads\Ukraine_Black_Sea_2020_2025_Feb14.csv'
df= pd.read_csv(file_path)

# Check the data types and for missing values
print(df.info())

# Check for missing data percentages
missing_df = df.isnull().sum()
missing_percentage = (missing_df/len(df))*100
print(missing_percentage)

# Drop columns wid more than 90% of missing data
columns_to_drop = missing_percentage[missing_percentage > 90].index
data = df.drop(columns=columns_to_drop)

# Fill missing values for categorical columns with 'Unknown'
categorical_cols = data.select_dtypes(include=['object']).columns
data[categorical_cols] = data[categorical_cols].fillna('Unknown')

# Fill missing values for categorical columns with the median
numerical_cols = data.select_dtypes(include=[np.number]).columns
data[numerical_cols] =data[numerical_cols].fillna(data[numerical_cols].median())

# to check if there still missing values
print(data.isnull().sum())

# statistics
print(data.describe())

# check the distribution of event types
print(data["event_type"].value_counts())

# analyze fatalities distribution
print(data["fatalities"].describe())

# analyze event distribution by year
print(data["year"].value_counts())

#  Trend Analysis: Event Types Over Time
event_trends = data.groupby(['year', 'event_type']).size().unstack().fillna(0)

# plotting the trend
plt.figure(figsize=(10,6))
event_trends.plot(kind='line', marker='o')
plt.title('Event Type Trends Over the Years')
plt.xlabel('Year')
plt.ylabel('Number of events')
plt.xticks(event_trends.index, rotation=45)
plt.grid(True)
plt.show()

# Comparison of event types by region
event_region = data.groupby(['region', 'event_type']).size().unstack().fillna(0)

plt.figure(figsize=(10, 6))
event_region.plot(kind='bar', stacked=True)
plt.title('Event Type Distribution by Region')
plt.xlabel('Region')
plt.ylabel('Number of Events')
plt.grid(True)
plt.show()

# Average fatalities per event type
fatalities_by_event = data.groupby('event_type')['fatalities'].mean().sort_values(ascending=False)

plt.figure(figsize=(8, 5))
sns.barplot(x=fatalities_by_event.values, y=fatalities_by_event.index)
plt.title('Average Fatalities by Event Type')
plt.xlabel('Average Fatalities')
plt.ylabel('Event Type')
plt.grid(True)
plt.show()

# Export the cleaned dataset
data.to_csv('clean_data.csv', index=False)
