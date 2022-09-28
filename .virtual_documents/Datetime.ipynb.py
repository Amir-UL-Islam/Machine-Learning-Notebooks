# Libraries
import pandas as pd
from datetime import datetime


# Reading the data
data = pd.read_pickle('datasets/florida_hurricane_dates.pkl')
data_datetime = pd.read_csv('datasets/capital-onebike.csv')


display(data_datetime.head())
data_datetime.info()


onebike_datetimes = []
for i in data_datetime['Start date']:
    temp = dict()
    temp['Start'] = datetime.strptime(i, 'get_ipython().run_line_magic("Y-%m-%d", " %H:%M:%S')")
    for i in data_datetime['End date']:
        temp['End'] = datetime.strptime(i, 'get_ipython().run_line_magic("Y-%m-%d", " %H:%M:%S')")
        onebike_datetimes.append(temp)


# Create dictionary to hold results
trip_counts = {'AM': 0, 'PM': 0}
  
# Loop over all trips
for trip in onebike_datetimes:
  # Check to see if the trip starts before noon
    if trip['Start'].hour < 12:
    # Increment the counter for before noon
        trip_counts['AM'] += 1
    else:
    # Increment the counter for after noon
        trip_counts['PM'] += 1
print(trip_counts)


# Load CSV into the rides variable
rides = pd.read_csv('datasets/capital-onebike.csv', 
                    parse_dates = ['Start date', 'End date'])

# Print the initial (0th) row
print(rides.iloc[0])


# Subtract the start date from the end date
ride_durations = rides['End date'] - rides['Start date']

# Convert the results to seconds
rides['Duration'] = ride_durations.dt.total_seconds()

print(rides['Duration'].head())


# Create joyrides
joyrides = (rides['Start station'] == rides['End station'])

# Total number of joyrides
print("{} rides were joyrides".format(joyrides.sum()))

# Median of all rides
print("The median duration overall was {:.2f} seconds"\
      .format(rides['Duration'].median()))

# Median of joyrides
print("The median duration for joyrides was {:.2f} seconds"\
      .format(rides[joyrides]['Duration'].median()))


# Import matplotlib
import matplotlib.pyplot as plt

# Resample rides to daily, take the size, plot the results
rides.resample('D', on= 'Start date')\
  .size()\
  .plot(ylim = [0, 15]) # Use ylim = [0, 15] to Get or set the y-limits of the current axes.

# Show the results
plt.show()


# Import matplotlib
import matplotlib.pyplot as plt

# Resample rides to monthly, take the size, plot the results
rides.resample('M', on = 'Start date')\
  .size()\
  .plot(ylim = [0, 150])

# Show the results
plt.show()


# Resample rides to be monthly on the basis of Start date
monthly_rides = rides.resample('M', on='Start date')['Member type']
print(monthly_rides.value_counts(), ' \n')
print(monthly_rides.size(),' \n' )

# Take the ratio of the .value_counts() over the total number of rides
print(monthly_rides.value_counts() / monthly_rides.size())



# Group rides by member type, and resample to the month
grouped = rides.groupby('Member type')\
  .resample('M', on='Start date')

# Print the median duration for each group
print(grouped['Duration'].median())


# Localize the Start date column to America/New_York
rides['Start date'] = rides['Start date'].dt.tz_localize('America/New_York', ambiguous='NaT')

# Print first value
print(rides['Start date'].iloc[0])

# Convert the Start date column to Europe/London
rides['Start date'] = rides['Start date'].dt.tz_convert('Europe/London')

# Print the new value
print(rides['Start date'].iloc[0])


# Add a column for the weekday of the start of the ride
rides['Ride start weekday'] = rides['Start date'].dt.day_name()

# Print the median trip time per weekday
print(rides.groupby('Ride start weekday')['Duration'].median())


# Shift the index of the end date up one; now subract it from the start date
rides['Time since'] = rides['Start date'] - (rides['End date'].shift(1))

# Move from a timedelta to a number of seconds, which is easier to work with
rides['Time since'] = rides['Time since'].dt.total_seconds()

# Resample to the month
monthly = rides.resample('M', on = 'Start date')

# Print the average hours between rides each month
print(monthly['Time since'].mean()/(60*60))






