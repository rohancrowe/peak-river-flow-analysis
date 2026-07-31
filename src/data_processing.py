from pathlib import Path
import pandas as pd

### The aim of this section is to process the raw, incomplete time series data into a list of independent peaks
### A threshold is chosen so we're left with, on average, the 5 largest peaks per year

## Dealing with NaN data:

# This isolates the flood-risk-analysis folder saved locally
project_root = Path(__file__).resolve().parent.parent

# Path to the locally saved flood-risk-analysis folder
csv_path = project_root / "data" / "raw" / "lune_river_flow_data.csv"


df = pd.read_csv(csv_path)

# Remove all superfluous data - for ever river flow data csv file, all superfluous data has 3 entries per row, and the time series data has 2 per row
# Therefore this method works for other rivers' datasets

test_column = df.iloc[:, 2]
first_NaN_index = test_column.isna().idxmax()
time_series = df.iloc[first_NaN_index:]
time_series = time_series.iloc[:, :2] #removes NaN column
time_series.columns = ["date", "river flow (m3/s)"]

time_series["river flow (m3/s)"] = pd.to_numeric(time_series["river flow (m3/s)"], errors="coerce") # changes data from object to numeric
time_series["date"] = pd.to_datetime(time_series["date"])


# Looking at the structure of isna_list, all of 1977 and 1978 are missing data (and some more dates, see below). Therefore we need to leave them out of further analysis.
# First we must analyse the time series around the cut points to see if we need to remove any peaks.
isna_list = time_series[time_series["river flow (m3/s)"].isna()].index.tolist()

river_flow_mean = time_series["river flow (m3/s)"].mean() ## needs to be calculated before we artificially set values as 0

for i in isna_list:
    time_series.loc[i, "river flow (m3/s)"] = 0 ## so we can see the cut clearly later (it will look like a drop off)

# print(time_series.loc[20818, "date"]) ## There are two more NaN points at 2015-12-12 and 2015-12-13, we will come back to these

import matplotlib.pyplot as plt

# Overall time series

time_series["date"] = pd.to_datetime(time_series["date"])

time_series = time_series.set_index("date")

fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(time_series.index, time_series["river flow (m3/s)"], linewidth=0.8)

ax.set_title("River Flow Time Series")
ax.set_xlabel("date")
ax.set_ylabel("river flow (m³/s)")
ax.set_ylim(bottom=0)
# ax.set_xlim(time_series.index.min(), time_series.index.max())

ax.grid(True)

plt.tight_layout()

plt.savefig(project_root / "results" / "figures" / "overall_time_series.png", dpi=300)
plt.show()

# Time series around the NaN values

period_1977 = time_series.loc["1976-12-01":"1977-01-10"]
period_1978 = time_series.loc["1978-12-20":"1979-01-31"]

fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(period_1977.index, period_1977["river flow (m3/s)"])

ax.set_title("River Flow January 1977")
ax.set_xlabel("date")
ax.set_ylabel("river flow (m³/s)")

ax.axhline(y = river_flow_mean , color = "green" , linewidth = 1, label = "mean flow")
ax.axvline(x = pd.Timestamp("1977-01-01") , color = "red", linestyle = ":", linewidth = 1, label = "cut line")

ax.legend()

# plt.savefig(project_root / "results" / "figures" / "1977_cut_analysis.png", dpi=300)
# plt.show()

fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(period_1978.index, period_1978["river flow (m3/s)"])

ax.set_title("River Flow January 1979")
ax.set_xlabel("date")
ax.set_ylabel("river flow (m³/s)")

ax.axhline(y = river_flow_mean , color = "green" , linewidth = 1, label = "mean flow")
ax.axvline(x = pd.Timestamp("1978-12-31") , color = "red", linestyle = ":", linewidth = 1, label = "cut line")

ax.legend()

# plt.savefig(project_root / "results" / "figures" / "1978_cut_analysis", dpi=300)
# plt.show()


# Looking at the graphs, we don't have to remove any peaks around the cut lines
# We have to apply the same analysis as above to the endpoints of the dataset. 
# The last value in the dataset is below the mean, so like above, there is no peak to remove here
# However, the first value is a peak, see below

period_1959 = time_series.loc["1959-01-01":"1959-01-31"]

fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(period_1959.index, period_1959["river flow (m3/s)"])

ax.set_title("River Flow January 1959")
ax.set_xlabel("date")
ax.set_ylabel("river flow (m³/s)")

ax.axhline(y = river_flow_mean , color = "green" , linewidth = 1, label = "mean flow")

ax.legend()

# plt.savefig(project_root / "results" / "figures" / "1959_edge_analysis", dpi=300)
# plt.show()

# The trough here is at 1959-01-08, therefore we set these values to 0 to remove the peak

time_series.loc["1959-01-01":"1959-01-08", "river flow (m3/s)"] = 0

# Now we check the two NaN values in 2015

period_2015 = time_series.loc["2015-12-01":"2015-12-31"]

fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(period_2015.index, period_2015["river flow (m3/s)"])

ax.set_title("River Flow December 2015")
ax.set_xlabel("date")
ax.set_ylabel("river flow (m³/s)")

ax.axhline(y = river_flow_mean , color = "green" , linewidth = 1, label = "mean flow")
ax.axvline(x = pd.Timestamp("2015-12-12") , color = "red", linestyle = ":", linewidth = 1, label = "cut line 1")
ax.axvline(x = pd.Timestamp("2015-12-13") , color = "red", linestyle = ":", linewidth = 1, label = "cut line 2")

ax.legend()

# plt.savefig(project_root / "results" / "figures" / "2015_analysis.png", dpi=300)
# plt.show()

# Looking at the graph, the NaN values are extremely unlikely to have contributed to a peak so we can move on to independence tests


## 1. Finding peaks:

# A peak is a point that has lower values on either side. This finds all peaks by filtering by values greater than both adjacent values

import numpy as np

time_series = time_series.drop(time_series.loc["1977-01-01":"1978-12-31"].index)

# In the 64.75 years of data, we want roughly 5 peaks over threshold per year, so we want roughly 334 peaks
# The independence tests will ensure each peak is independent, we shouldn't worry if some years have more peaks than others, some years have more rain than others
# First we calculate mean time to rise with all the single peaks we can get
# For single peaks we use the condition that both the trough before and the trough after must be below the mean flow

ts = time_series.copy().sort_index()

if ts.index.name is None:
    ts = ts.reset_index().rename(columns={"index": "date"}) # makes sure when we reset index and pull the dates into a new column, it has the column name of "date"
else:
    ts = ts.reset_index().rename(columns={ts.index.name: "date"})

ts["date"] = pd.to_datetime(ts["date"]) 
ts = ts.sort_values("date").reset_index(drop=True)


flow = ts["river flow (m3/s)"]

peak_helper = (flow > flow.shift(1)) & (flow >= flow.shift(-1)) # finds peaks and troughs
trough_helper = (flow < flow.shift(1)) & (flow <= flow.shift(-1))

peaks = ts.loc[peak_helper, ["date", "river flow (m3/s)"]].copy()
troughs = ts.loc[trough_helper, ["date", "river flow (m3/s)"]].copy()

peaks = peaks.rename(columns={"date": "peak_date", "river flow (m3/s)": "peak_flow"}).sort_values("peak_date") # we create peak and trough dataframes ordered chronologically
troughs = troughs.rename(columns={"date": "trough_date", "river flow (m3/s)": "trough_flow"}).sort_values("trough_date")

candidate_peaks = peaks.nlargest(500, "peak_flow").sort_values("peak_date").reset_index(drop=True) # we want the largest 334 peaks in the end, so we hope 500 is enough that once filtered, we have more than 334, then we sort chronologically
troughs = troughs.reset_index(drop=True)


trough_dates = pd.DatetimeIndex(troughs["trough_date"])

before_dates = []
before_flows = []
after_dates = []
after_flows = []

for peak_date in candidate_peaks["peak_date"]:
    pos = trough_dates.searchsorted(peak_date) # we look for where the peak date fits into the trough dates

    # nearest trough before the peak
    if pos > 0:
        before_row = troughs.iloc[pos - 1]
        before_dates.append(before_row["trough_date"])
        before_flows.append(before_row["trough_flow"])
    else:
        before_dates.append(pd.NaT)
        before_flows.append(np.nan)

    # nearest trough after the peak
    if pos < len(troughs):
        after_row = troughs.iloc[pos]
        after_dates.append(after_row["trough_date"])
        after_flows.append(after_row["trough_flow"])
    else:
        after_dates.append(pd.NaT)
        after_flows.append(np.nan)

result = candidate_peaks.copy() # compile this all into a dataframe
result["trough_before_date"] = before_dates
result["trough_before_flow"] = before_flows
result["trough_after_date"] = after_dates
result["trough_after_flow"] = after_flows

result["time_to_previous_trough"] = result["peak_date"] - result["trough_before_date"]
result["time_to_next_trough"] = result["trough_after_date"] - result["peak_date"]

result = result[result["trough_before_date"].notna() & result["trough_after_date"].notna() & (result["trough_before_date"] < result["peak_date"]) & (result["trough_after_date"] > result["peak_date"])].copy()

single_peaks_results  = result[result["trough_before_flow"].notna() & result["trough_after_flow"].notna() & (result["trough_before_flow"] < river_flow_mean) & (result["trough_after_flow"] < river_flow_mean)]

mean_rise_time = single_peaks_results["time_to_previous_trough"].mean()

# Below is to justify how we calculate mean time to rise, see README.md
# period = time_series.loc["2010-01-01":"2010-12-31"]
# fig, ax = plt.subplots(figsize=(12, 6))
# ax.plot(period.index, period["river flow (m3/s)"])
# ax.set_title("River Flow 2010")
# ax.set_xlabel("date")
# ax.set_ylabel("river flow (m³/s)")
# ax.axhline(y = river_flow_mean , color = "green" , linewidth = 1, label = "mean flow")
# ax.legend()
# ax.set_ylim(bottom=0)
# ax.set_xlim(pd.Timestamp("2010-01-01"), pd.Timestamp("2010-12-31"))
# ax.grid(True)

# plt.savefig(project_root / "results" / "figures" / "2010.png", dpi=300)
# plt.show()


## We test for independence

test_1 = result.sort_values("peak_date").copy()

# Start a new group whenever the gap from the previous peak is too large
test_1["gap_from_prev_peak"] = test_1["peak_date"].diff() 
test_1["event_group_1"] = (test_1["gap_from_prev_peak"] > 3 * mean_rise_time).cumsum() # we group each independence window by positive integers

keep_index = test_1.groupby("event_group_1")["peak_flow"].idxmax() # we keep the highest peak in this window
test_1_results = test_1.loc[keep_index].sort_values("peak_date").reset_index(drop=True) # apply the above filter to our results dataframe

min_flow_between = []

for i in range(len(test_1_results) - 1):
    start = test_1_results.loc[i, "peak_date"] # create a time series window between this peak and the next peak where we'll look for the minimum flow
    end = test_1_results.loc[i + 1, "peak_date"]

    segment = time_series.loc[(time_series.index > start) & (time_series.index < end), "river flow (m3/s)"]

    min_flow_between.append(segment.min()) # find the minimum flow between two peaks and add to our new column
min_flow_between.insert(0, 0)

test_1_results["min_flow_between_next_peak"] = min_flow_between

test_1_results["event_group_2"] = (test_1_results["min_flow_between_next_peak"] < (2/3) * test_1_results["peak_flow"]).cumsum()
test_1_results["event_group_2"] = (test_1_results["min_flow_between_next_peak"] < (2/3) * test_1_results["peak_flow"]).shift(1, fill_value=False).cumsum()

keep_index_2 = test_1_results.groupby("event_group_2")["peak_flow"].idxmax()
test_2_results = test_1_results.loc[keep_index_2].sort_values("peak_date").reset_index(drop=True) 

# Our arbritary choosing of the largest 500 peaks earlier so we don't have to iterate over all peaks has actually worked out very well, after filtering there are 335 remaining

sorted_peaks = test_2_results.sort_values("peak_flow", ascending=False).reset_index(drop=True)

peak_334 = sorted_peaks.loc[333, "peak_flow"]
peak_335 = sorted_peaks.loc[334, "peak_flow"]

POT_threshold = (peak_334 + peak_335) / 2

processed_data = test_2_results.nlargest(334, "peak_flow").sort_values("peak_date").reset_index(drop=True)

processed_data = processed_data[["peak_date", "peak_flow"]]

# processed_data.to_csv("flood-risk-analysis/data/processed/independent_peaks.csv", index=False)

# pd.DataFrame({"POT_threshold": [POT_threshold]}).to_csv("flood-risk-analysis/data/processed/POT_threshold.csv", index=False)


