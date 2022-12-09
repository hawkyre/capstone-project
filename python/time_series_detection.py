import numpy as np


def detect_local_trends(time_series):
    # Initialize empty dictionary to store maximum trend values for each starting point
    max_trends = {}

    # Initialize empty list to store detected trends
    local_trends = []

    # Loop through the time series to find local trends
    for i in range(len(time_series)):
        # Loop through subarrays of increasing size starting at the current point
        for j in range(i+1, len(time_series)+1):
            # Calculate the trend across the subarray using the least squares method
            subarray = time_series[i:j]
            x_values = list(range(i, j))
            y_values = subarray
            A = np.vstack([x_values, np.ones(len(x_values))]).T
            trend, _ = np.linalg.lstsq(A, y_values, rcond=None)[0]

            # Check if the trend is positive or negative
            if trend > 0:
                # Trend is positive, check if it is the largest trend for the starting point
                if i not in max_trends or trend > max_trends[i][1]:
                    max_trends[i] = ("positive", trend)
                    local_trends.append(("positive", i, j, trend))
            elif trend < 0:
                # Trend is negative, check if it is the largest trend for the starting point
                if i not in max_trends or trend < max_trends[i][1]:
                    max_trends[i] = ("negative", trend)
                    local_trends.append(("negative", i, j, trend))

    # Return the list of detected local trends
    return local_trends
