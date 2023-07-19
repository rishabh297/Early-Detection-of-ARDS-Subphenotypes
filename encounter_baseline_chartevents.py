import pandas as pd
import numpy as np
from fancyimpute import IterativeImputer
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from copy import deepcopy
from collections import deque
from sklearn.cluster import DBSCAN
import hdbscan
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import ParameterGrid
from scipy.stats import shapiro
from scipy import stats

# Load data
encounter_events = pd.read_csv('/Users/rishabhgoel/Desktop/Gen1E-RIDGE/MIMIC-IV Project and eICU-CRD/mimic-iv-2.2/icu/encounter_chartevents.csv')

# Load admissions data
admission_df = pd.read_csv('/Users/rishabhgoel/Desktop/Gen1E-RIDGE/MIMIC-IV Project and eICU-CRD/mimic-iv-2.2/hosp/admissions.csv')


# Merge encounter_events and admission_df
merged_df = pd.merge(encounter_events, admission_df[['subject_id', 'hadm_id', 'admittime']], on=['subject_id', 'hadm_id'], how='inner')

# Convert 'admittime' to datetime
merged_df['admittime'] = pd.to_datetime(merged_df['admittime'])
merged_df['charttime'] = pd.to_datetime(merged_df['charttime'])


transfers = pd.read_csv('/Users/rishabhgoel/Desktop/Gen1E-RIDGE/MIMIC-IV Project and eICU-CRD/mimic-iv-2.2/hosp/transfers.csv')

# Ensure 'intime' is in datetime format
transfers['intime'] = pd.to_datetime(transfers['intime'])

icu_units = [
    'Medical Intensive Care Unit (MICU)',
    'Surgical Intensive Care Unit (SICU)',
    'Medical/Surgical Intensive Care Unit (MICU/SICU)',
    'Cardiac Vascular Intensive Care Unit (CVICU)',
    'Coronary Care Unit (CCU)',
    'Neuro Surgical Intensive Care Unit (Neuro SICU)',
    'Trauma SICU (TSICU)'
]

# Filter for ICU admissions
transfers_icu = transfers[transfers['careunit'].isin(icu_units)]

# Sort by 'intime'
transfers_icu = transfers_icu.sort_values('intime')

# Group by 'subject_id' and 'hadm_id' and take the first 'intime'
first_icu_intimes = transfers_icu.groupby(['subject_id', 'hadm_id'])['intime'].first().reset_index()

# Merge the results back into the original dataframe
merged_df = pd.merge(merged_df, first_icu_intimes[['subject_id', 'hadm_id', 'intime']], on=['subject_id', 'hadm_id'], how='inner')


# Calculate the time difference in hours
merged_df['time_diff_hours'] = (merged_df['charttime'] - merged_df['intime']).dt.total_seconds() / 3600

# Filter the DataFrame to include only events within 24 hours of admission
merged_df = merged_df[merged_df['time_diff_hours'] <= 24]


# Perform similar operations as for day 1 (pivot, merge etc.),
# but with data from 'merged_df_day3'. This could be put into a function for code reusability.

encounter_events = merged_df.drop(columns = ['time_diff_hours'])


item_details_df = pd.read_csv('/Users/rishabhgoel/Desktop/Gen1E-RIDGE/MIMIC-IV Project and eICU-CRD/mimic-iv-2.2/icu/d_items.csv')
item_details_df = item_details_df[(item_details_df['param_type'] == 'Numeric')]

#  & (item_details_df['linksto'] == 'chartevents')

item_id_dict = dict(zip(item_details_df['itemid'], item_details_df['abbreviation']))

item_ids = list(item_id_dict.keys())

pao2_var_NIH  = 'PaO2'
pao2_var_all  = 'PO2 (Arterial)'

weight_nih = 'Weight'
weight_all = 'Daily Weight'

height_nih = 'Height'
height_all = 'Height (cm)'


# Define item IDs
item_ids = [223762, 220052, 220045, 226540, 220545, 220546,
227457, 220645, 229761, 220615, 226537, 220621, 227456, 225690, 220224, 220277, 223835, 220235, 220181,
 225698, 223679, 225624]


Vent_params = [224685,224687, 224690, 224695, 220339, 227287]
random_params = [227073, 227443, 224639, 226707]

# Filter data
filtered_encounter_events = encounter_events[encounter_events['itemid'].isin(item_ids)]
filtered_encounter_events['charttime'] = pd.to_datetime(filtered_encounter_events['charttime'])

# Find rows with minimum charttime
idx = filtered_encounter_events.groupby(['subject_id', 'hadm_id', 'itemid'])['charttime'].idxmin()
filtered_encounter_events = filtered_encounter_events.loc[idx]

# Pivot DataFrame
pivot_df = filtered_encounter_events.pivot_table(index=['subject_id', 'hadm_id'], columns='itemid', values='value')
pivot_df.reset_index(inplace=True)

# Define item ID dictionary
item_id_dict = {
    # 224639: "Weight",
    # 226707: "Height",
    223762: "Temperature",
    220052: "Blood Pressure (Mean)",
    220045: "Heart Rate",
    # 224685: "Tidal Volume (observed)",
    # 224687: "Total Minute Ventilation",
    # 224690: "Respiratory Rate",
    226540: "Hematocrit (whole blood)",
    220545: "Hematocrit (serum)",
    220546: "White Blood Cells",
    227457: "Platelets",
    220645: "Sodium",
    229761: "Creatinine (whole blood)",
    220615: "Creatinine (serum)",
    226537: "Glucose (whole blood)",
    220621: "Glucose (serum)",
    227456: "Albumin",
    225690: "Bilirubin",
    220224: "PaO2",
    223835: "FiO2",
    220277: "SpO2",
    # 220339: "PEEP",
    # 227287: "HNFO",
    # 224695: "Peak Insp. Pressure",
    220235: "PCO2",
    220181: "Non Invasive Blood Pressure mean",
    225698:	'TCO2 (calc) Arterial',
    223679:	'TCO2 (calc) Venous',
    # 227073:	"Anion gap",
    225624:	"BUN",
    # 227443: 'HCO3 (serum)'
}

# Rename columns
pivot_df.rename(columns=item_id_dict, inplace=True)

# Identify unique item IDs in the data
unique_item_ids_in_data = filtered_encounter_events['itemid'].unique()

# Identify missing item IDs
missing_item_ids = set(item_id_dict.keys()) - set(unique_item_ids_in_data)

# Map missing item IDs back to names
missing_item_names = {item_id: item_id_dict[item_id] for item_id in missing_item_ids}

# Print missing items
print("The following items are missing from the data:")
for item_id, item_name in missing_item_names.items():
    print(f"Item ID: {item_id}, Item Name: {item_name}")

# Calculate BMI, first convert height from inches to meters
# pivot_df['BMI'] = pivot_df[weight_nih] / ((pivot_df[height_nih]/100)**2)

# Calculate PaO2/FiO2 and SpO2/FiO2, FiO2 divided by 100
pivot_df['PaO2/FiO2'] = pivot_df[pao2_var_NIH] / (pivot_df['FiO2'] / 100)
pivot_df['SpO2/FiO2'] = pivot_df['SpO2'] / (pivot_df['FiO2'] / 100)

pivot_df = pivot_df.drop(columns=[pao2_var_NIH, 'FiO2', 'SpO2/FiO2', 'PaO2/FiO2'])


# Load admissions data
admission_df = pd.read_csv('/Users/rishabhgoel/Desktop/Gen1E-RIDGE/MIMIC-IV Project and eICU-CRD/mimic-iv-2.2/hosp/admissions.csv')

# Merge hospital_stay_duration_df with expired_admissions
pivot_df = pivot_df.merge(admission_df[['subject_id', 'hadm_id', 'admittime', 'hospital_expire_flag']], on=['subject_id', 'hadm_id'], how='inner')

# Load patients data
patients_df = pd.read_csv('/Users/rishabhgoel/Desktop/Gen1E-RIDGE/MIMIC-IV Project and eICU-CRD/mimic-iv-2.2/hosp/patients.csv')

# Merge hospital_stay_duration_df with expired_admissions
pivot_df = pivot_df.merge(patients_df[['subject_id', 'dod', 'anchor_age']], on=['subject_id'], how='inner')

# # Define a mapping dictionary
# gender_dict = {'F': 0, 'M': 1}
#
# # Apply the map to the gender column
# pivot_df['gender'] = pivot_df['gender'].map(gender_dict)

# Convert 'admittime' to datetime
pivot_df['admittime'] = pd.to_datetime(pivot_df['admittime'])

# Calculate time until death
pivot_df['dod'] = pd.to_datetime(pivot_df['dod'])
pivot_df['time_until_death'] = (pivot_df['dod'] - pivot_df['admittime']).dt.days

copy_for_time_until_death = pivot_df

# Create a new column 'mortality_30d' where if 'days_until_death' <= 30 it's True, else it's False
pivot_df['mortality_30d'] = np.where(pivot_df['time_until_death'] <= 30, 1, 0)

# Create a new column 'mortality_90d' where if 'days_until_death' <= 30 it's True, else it's False
pivot_df['mortality_90d'] = np.where(pivot_df['time_until_death'] <= 90, 1, 0)

low = .01
high = .99

quant_df = pivot_df.drop(columns=['subject_id', 'hadm_id', 'admittime', 'dod','time_until_death']).quantile([low, high])

for name in list(pivot_df.drop(columns=['subject_id', 'hadm_id', 'admittime', 'dod','time_until_death']).columns):
    if pd.api.types.is_numeric_dtype(pivot_df.drop(columns=['subject_id', 'hadm_id', 'admittime', 'dod','time_until_death'])[name]):
        pivot_df.loc[pivot_df[name] > quant_df.loc[high, name], name] = np.nan
        pivot_df.loc[pivot_df[name] < quant_df.loc[low, name], name] = np.nan


 # Calculate the number of missing values and the corresponding percentage for each variable
missing_values_count = pivot_df.isnull().sum()
missing_values_percent = ((missing_values_count / len(pivot_df)) * 100).round(2)

# Combine the count and percentage in the required format
missing_values_formatted = missing_values_count.astype(str) + ' (' + missing_values_percent.astype(str) + '%)'

# Calculate the median and IQR for each variable
medians = pivot_df.drop(columns=['subject_id', 'hadm_id', 'admittime', 'dod','time_until_death']).median().round(2)
q1 = pivot_df.drop(columns=['subject_id', 'hadm_id', 'admittime', 'dod', 'time_until_death']).quantile(0.25).round(2)
q3 = pivot_df.drop(columns=['subject_id', 'hadm_id', 'admittime', 'dod', 'time_until_death']).quantile(0.75).round(2)

# Calculate mean and standard deviation
means = pivot_df.drop(columns=['subject_id', 'hadm_id', 'admittime', 'dod','time_until_death']).mean().round(2)
std_dev = pivot_df.drop(columns=['subject_id', 'hadm_id', 'admittime', 'dod','time_until_death']).std().round(2)

# Calculate the minimum and maximum for each variable
minimums = pivot_df.drop(columns=['subject_id', 'hadm_id', 'admittime', 'dod', 'time_until_death']).min().round(2)
maximums = pivot_df.drop(columns=['subject_id', 'hadm_id', 'admittime', 'dod', 'time_until_death']).max().round(2)

# Prepare two formatted values
medians_and_iqr_values_formatted = medians.astype(str) + ' (' + q1.astype(str) + '-' + q3.astype(str) + ')'
mean_and_stddev_values_formatted = means.astype(str) + '±' + std_dev.astype(str)
min_and_max_values_formatted = minimums.astype(str) + ' - ' + maximums.astype(str)

# Identify normal and non-normal distributions
is_normal = {column: (shapiro(pivot_df[column].dropna())[1] > 0.01) for column in pivot_df.drop(columns=['subject_id', 'hadm_id', 'admittime', 'dod','time_until_death']).columns}

# Apply formatting based on normality of distribution
baseline_values = {
    column: mean_and_stddev_values_formatted[column]
    if is_normal[column] else medians_and_iqr_values_formatted[column]
    for column in is_normal.keys()
}

# For 'mortality_30d' and 'hospital_expire_flag', overwrite the baseline value with mean and standard deviation
# baseline_values['gender'] = mean_and_stddev_values_formatted['gender']
baseline_values['mortality_30d'] = mean_and_stddev_values_formatted['mortality_30d']
baseline_values['mortality_90d'] = mean_and_stddev_values_formatted['mortality_90d']
baseline_values['hospital_expire_flag'] = mean_and_stddev_values_formatted['hospital_expire_flag']
baseline_values['anchor_age'] = mean_and_stddev_values_formatted['anchor_age']

# Create the final dataframe for the baseline stats
baseline_stats = pd.DataFrame({'Missing Values': missing_values_formatted,
                               'Baseline Value': pd.Series(baseline_values)})

baseline_stats.drop(['subject_id', 'hadm_id', 'time_until_death'])

baseline_stats.to_csv('/Users/rishabhgoel/Desktop/Gen1E-RIDGE/MIMIC-IV Project and eICU-CRD/mimic-iv-2.2/icu/baseline_everything_final_paper.csv')


statistics_df = pivot_df.drop(columns=['subject_id', 'hadm_id', 'admittime', 'dod']).describe().T

# Calculate missing data percentage before removal
missing_data_percentage = (pivot_df.isnull().sum() / len(pivot_df)) * 100
print("Missing data percentages before removal:")
print(missing_data_percentage.head(60))

missing_data_percentage = (pivot_df.isnull().sum() / len(pivot_df)) * 100
print(missing_data_percentage.head(60))
missing_threshold = 0.10
missing_data_percentage = pivot_df.isnull().mean()
pivot_df = pivot_df.loc[:, missing_data_percentage <= missing_threshold]

# Calculate missing data percentage after removal
missing_data_percentage_after = (pivot_df.isnull().sum() / len(pivot_df)) * 100
print("Missing data percentages after removal:")
print(missing_data_percentage_after.head(60))

# Compare before and after to find removed variables
removed_vars_due_to_missing_data = set(missing_data_percentage.index) - set(missing_data_percentage_after.index)
print("Variables removed due to high percentage of missing data:")
print(removed_vars_due_to_missing_data)

SEED = 11
# Initiate the MICE Imputer
mice_imputer = IterativeImputer(random_state=SEED)



mice_df = pivot_df.drop(columns=['admittime', 'hospital_expire_flag', 'mortality_30d', 'mortality_90d'])


# Compute statistics before imputation
mean_before = mice_df.drop(columns=['subject_id', 'hadm_id']).mean()
std_before = mice_df.drop(columns=['subject_id', 'hadm_id']).std()
median_before = mice_df.drop(columns=['subject_id', 'hadm_id']).median()

# Compute statistics before imputation
stats_before = mice_df.describe().transpose()

# Fit and transform the dataframe to fill NaN values
pivot_df_mice = mice_imputer.fit_transform(mice_df)
pivot_df_mice = pd.DataFrame(pivot_df_mice, columns=mice_df.columns)

# Compute statistics before imputation
mean_after = pivot_df_mice.drop(columns=['subject_id', 'hadm_id']).mean()
std_after = pivot_df_mice.drop(columns=['subject_id', 'hadm_id']).std()
median_after = pivot_df_mice.drop(columns=['subject_id', 'hadm_id']).median()

# Concatenate the stats
compare_stats = pd.concat([mean_before, mean_after, std_before, std_after, median_before, median_after], axis=1,
                          keys=['Mean Before Imputation', 'Mean After Imputation', 'Std Before Imputation', 'Std After Imputation','Median Before Imputation', 'Median After Imputation'])

# Concatenate the stats
final_stats = pd.concat([mean_before, std_before, median_before], axis=1,
                          keys=['Mean', 'Standard Deviation', 'Median'])

# Filter missing_data_percentage to only include the variables present in the compare_stats DataFrame
missing_data_percentage_filtered = missing_data_percentage[final_stats.index]

# Add the missing_data_percentage_filtered to the compare_stats DataFrame
final_stats['Missing Percentage'] = missing_data_percentage_filtered

final_stats.to_csv('/Users/rishabhgoel/Desktop/Gen1E-RIDGE/MIMIC-IV Project and eICU-CRD/mimic-iv-2.2/hosp/Baseline_Global_Patient_Stats.csv')

print(len(pivot_df_mice.columns))

def remove_collinear_features(x, threshold):
    '''
    Objective:
        Remove collinear features in a dataframe with a correlation coefficient
        greater than the threshold. Removing collinear features can help a model
        to generalize and improves the interpretability of the model.

    Inputs:
        x: features dataframe
        threshold: features with correlations greater than this value are removed

    Output:
        dataframe that contains only the non-highly-collinear features
    '''

    # Calculate the correlation matrix
    corr_matrix = x.corr()
    iters = range(len(corr_matrix.columns) - 1)
    drop_cols = []

    # For each column, check if there exist other highly correlated features
    for i in iters:
        for j in range(i):
            item = corr_matrix.iloc[j:(j+1), (i+1):(i+2)]
            col = item.columns
            row = item.index
            val = abs(item.values)

            # If correlation exceeds the threshold
            if val >= threshold:
                # Check if this column is already set to be dropped
                if col[0] not in drop_cols:
                    # If both features correlate with another feature, drop the one that
                    # has a larger mean correlation coefficient
                    mean_corr_col = abs(corr_matrix[col[0]]).mean()
                    mean_corr_row = abs(corr_matrix[row[0]]).mean()
                    if mean_corr_col > mean_corr_row:
                        drop_cols.append(col[0])
                    else:
                        drop_cols.append(row[0])

    # Drop the features with high correlation
    x = x.drop(drop_cols, axis=1)

    return x

# Record the column names before removal
cols_before = set(pivot_df_mice.columns)

# calculate the correlation matrix
corr = pivot_df_mice.drop(columns=['subject_id', "hadm_id"]).corr()

# plot the heatmap
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, cmap='RdBu')

# display the plot
plt.show()

# Apply the function to your data
pivot_df_mice = remove_collinear_features(pivot_df_mice, 0.5)

# Record the column names after removal
cols_after = set(pivot_df_mice.columns)

# Find removed variables
removed_vars_due_to_collinearity = cols_before - cols_after
print("Variables removed due to high collinearity:")
print(removed_vars_due_to_collinearity)

# Check if a row contains any negative values and keep the rows that don't
pivot_df_mice = pivot_df_mice[(pivot_df_mice >= 0).all(axis=1)]

# Merge hospital_stay_duration_df with expired_admissions
pivot_df_unscaled = pivot_df_mice.merge(pivot_df[['subject_id', 'hadm_id', 'hospital_expire_flag', 'mortality_30d', 'mortality_90d']], on=['subject_id', 'hadm_id'], how='inner')

pivot_df_unscaled = pivot_df_unscaled.merge(copy_for_time_until_death[['subject_id', 'hadm_id', 'time_until_death']], on=['subject_id', 'hadm_id'], how='inner')

pivot_df_unscaled.to_csv('/Users/rishabhgoel/Desktop/Gen1E-RIDGE/MIMIC-IV Project and eICU-CRD/mice_filled_baseline_preprocessed_out.csv', index=False)


print(pivot_df_mice.columns)

# Add a small constant to avoid log of zero
constant = 1e-10

# Perform log transformation
log_transformed_df = np.log(pivot_df_mice.drop(columns=['subject_id', 'hadm_id']) + constant)

# Standardize the data
scaler = StandardScaler()
pivot_df_scaled = scaler.fit_transform(log_transformed_df)
pivot_df_scaled = pd.DataFrame(pivot_df_scaled, columns=(pivot_df_mice.drop(columns=['subject_id', 'hadm_id'])).columns)

pivot_df_scaled.to_csv('/Users/rishabhgoel/Desktop/Gen1E-RIDGE/MIMIC-IV Project and eICU-CRD/mice_filled_baseline_preprocessed_scaled_out.csv', index=False)

print(pivot_df_scaled)
