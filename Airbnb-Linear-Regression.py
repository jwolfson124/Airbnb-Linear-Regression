#!/usr/bin/env python
# coding: utf-8

# In[1]:


##import the entire dataset in a way where we can just add the next file in with no issues
#import libraries and packages
import kagglehub
import glob
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
import streamlit as st
import matplotlib as plt
import altair as alt
from datetime import date, datetime
import re
import ast
from sklearn.preprocessing import StandardScaler
import random
from statsmodels.stats.outliers_influence import variance_inflation_factor
from pathlib import Path
st.set_page_config(layout='wide') #make sure we can use the entire streamlit page


# ## Bring in the data from insideairbnb.com and use the listings.csv.gz

# In[672]:


# Path to the "Air BnB Data" folder inside the repo
DATA_DIR = Path(__file__).parent / "Air BnB Data"

# Get all .xls files in that folder
excel_files = list(DATA_DIR.glob("*.xls"))

# Step 1: Build list of all unique columns across files
all_columns = set()
for file_path in excel_files:
    df_head = pd.read_excel(file_path, nrows=1, engine="xlrd")
    all_columns.update(df_head.columns)

all_columns = list(all_columns)

# Step 2: Load data from each file, align columns, and add source column
dfs = []
missing_column_report = []

for file_path in excel_files:
    df = pd.read_excel(file_path, engine="xlrd")
    original_cols = set(df.columns)
    missing_cols = list(set(all_columns) - original_cols)

    df = df.reindex(columns=all_columns)
    df["source_file"] = file_path.name
    dfs.append(df)

    if missing_cols:
        missing_column_report.append({
            "file": file_path.name,
            "missing_columns": sorted(missing_cols)
        })

# Step 3: Combine all into one DataFrame
combined_df = pd.concat(dfs, ignore_index=True)

# Step 4: Optional report
report_df = pd.DataFrame(missing_column_report)

print(f"Loaded {len(excel_files)} files from {DATA_DIR}")
print("Combined shape:", combined_df.shape)
print(report_df)
#st.write("Data for App Loaded")


# In[ ]:





# In[3]:


# # Step 1: Set the folder path
# folder_path = "/Users/student/Desktop/Dashboard Work/Linear Model House Pricing/Air BnB Data"
# excel_files = glob.glob(os.path.join(folder_path, "*.xls"))

# # Step 2: Find all unique columns across the files
# all_columns = set()
# file_columns_map = {}

# for file in excel_files:
#     df = pd.read_excel(file, nrows=1)  # Read header only
#     file_columns_map[file] = set(df.columns)
#     all_columns.update(df.columns)

# all_columns = list(all_columns)

# # Step 3: Load data and align all columns
# dfs = []
# missing_column_report = []

# for file in excel_files:
#     df = pd.read_excel(file)
#     original_cols = set(df.columns)
#     missing_cols = list(set(all_columns) - original_cols)

#     # Reindex with all columns so missing ones are filled with NaN
#     df = df.reindex(columns=all_columns)

#     # Optional: add a column to indicate which file the data came from
#     df['source_file'] = os.path.basename(file)
#     dfs.append(df)

#     # Track which columns were missing in this file
#     if missing_cols:
#         missing_column_report.append({
#             'file': os.path.basename(file),
#             'missing_columns': missing_cols
#         })

# # Step 4: Combine all into one large DataFrame
# combined_df = pd.concat(dfs, ignore_index=True)

# # Step 5: Create and print report of missing columns
# report_df = pd.DataFrame(missing_column_report)


# In[ ]:





# ## Identify columns that are not consistent and remove them from df

# In[5]:


#columns to drop

column_drop = set()

for i, row in report_df.iterrows():
    #print("Missing rows from ", row['file'])
    #print(row['missing_columns'])
    for j in range(len(row['missing_columns'])):
        column_drop.add(row['missing_columns'][j])
        print(row['missing_columns'][j])
#make a copy of the combined_df called df
df = combined_df.copy()

df = df.drop(columns=column_drop)#, inplace=True)

#these numbers should reflect removing columns that are not in all of the datasets
#print(len(df.columns))
#print(len(combined_df.columns))


# In[ ]:





# # Identify Columns that will not be useful to the algorythm

# In[7]:


#remove URL
keep_columns = ['host_since', 'host_response_rate', 'host_acceptance_rate',
                'host_is_superhost', 'host_listings_count', 'host_total_listings_count',
               'host_identity_verified', 'neighbourhood_cleansed', 'room_type', 'accommodates', 
               'bathrooms', 'bedrooms', 'beds', 'amenities', 'minimum_nights', 'maximum_nights', 'has_availability',
               'availability_365', 'calendar_last_scraped',
                'number_of_reviews',
                'review_scores_rating', 'review_scores_cleanliness', 'review_scores_checkin',
                'review_scores_communication', 'review_scores_value', 'instant_bookable', 
                'calculated_host_listings_count', 'reviews_per_month', 'price','source_file'
               ]

small_df = df[keep_columns].copy()

original_df = df.copy()

#change columns: 
#host_is_superhost - binary, 
#host_has_profile_pic - binary, 
#host_identity_verified - binary
#amenities - turn the different amenities into yes and no
#has availability - somewhat binary

#change any datetime columns

#opportunities for improvement: 
#enter in latitude and longitude for more specific location pricing
#currently just using neighbourhood_cleansed

#st.write('Dataset Created')


# ## Change any datetime columns to integer values

# In[9]:


#columns that need to be changed
print(small_df.select_dtypes(include=['datetime', 'datetime64[ns]', 'datetimetz']).columns)

#change host since to get total days as a host, also remove the NAN values, and drop the old column
small_df = small_df[~small_df['host_since'].isna()]
small_df['host_since_years'] = (datetime.now() - small_df['host_since']).dt.days.astype(int) / 365
small_df = small_df.drop(columns='host_since')


#turn the small_df calendar last scraped into the month and the year as a string
small_df['calendar_last_scraped'] = small_df['calendar_last_scraped'].dt.strftime('%B - %Y')



# In[550]:





# ## Change categorical values into dummy variables

# In[11]:


df = small_df.copy()

#display(df.select_dtypes(include=['object']))

#turn true into 1 and all else into 0
df['host_is_superhost'] = df['host_is_superhost'].apply(lambda x: 1 if x == 't' else 0)
df['host_identity_verified'] = df['host_identity_verified'].apply(lambda x: 1 if x == 't' else 0)
df['has_availability'] = df['has_availability'].apply(lambda x: 1 if x == 't' else 0)
df['instant_bookable'] = df['instant_bookable'].apply(lambda x: 1 if x == 't' else 0)

#display(df.select_dtypes(include=['object']))
df['city'] = df['source_file'].apply(lambda x: re.search(r"^[^\d]+", x).group().strip())
df = df.drop(columns='source_file')

print(df['amenities'])


# In[23]:


#df.columns


# ## Use Total Number of Amenities Instead of Individual

# In[15]:


#the ast.literal_eval turns the string that holds a list into just a list of the different amenities
#the lambda(x: ','.join(x)) is going to then take the list of strings and turn them into one long list
#this will allow the string to be turned into dummies using the "," seperator
df['amenities'] = df['amenities'].apply(ast.literal_eval).apply(lambda x: ','.join(x))

#df_dummies = df['amenities'].str.get_dummies(sep=',')
#df_dummies


# In[17]:


def count_amenities(amenities_str):
    #split the amenities into a list
    amenities_list = amenities_str.split(',')
    
    #remove any whitespace that might have been weirdly seperated
    amenities_list = [item.strip() for item in amenities_list]

    #turn the list into a set to make sure there are no duplicates
    unique_amenities = set(amenities_list)
    
    return len(unique_amenities)

#change the amenities list into a count of all amenities
df['amenities'] = df['amenities'].apply(count_amenities)

#st.write("Amenities Transformation Complete")


# ## edit the data so that it will show entire vs shared vs private room as opposed to all the different options

# In[2208]:


#df['room_type']


# ## Get dummy values and apply prefix to help with organizatioon

# In[19]:


#create a list of dummy columns
dummy_cols = df.select_dtypes(include=['object']).columns

#identify what the prefix for each of the columns will be
pattern = re.compile(r'^[^_]+')
prefix = [pattern.match(c).group(0) for c in dummy_cols]

#get the dummys and apply the prefixs
dummy_values = pd.get_dummies(df[dummy_cols],prefix=prefix, dtype='uint8', sparse=True)

df = pd.concat([df, dummy_values], axis=1).drop(columns=dummy_cols)


# In[21]:


timeline_cols = df.columns[df.columns.str.contains('calendar')]
#timeline_cols


# In[ ]:





# ## identify missing data and how to deal with it the means, medians, max, and min to understand how similar the information is

# In[23]:


#remove all instances of missing price
df = df[df['price'].notna()]


#identify what columns have .isna() and how we want to deal with them
missing_cols = df.columns[df.isna().any()].tolist()
        

#see how the means change throughout the dataset by quarter
timeline_cols = df.columns[df.columns.str.contains('calendar')]

#create a blank list that will hold all information and be turned into dataframe later

missing_data_table = []


#review scores rating are all missing close to the same amount
for quarter in timeline_cols:
    quarter_mask = df[quarter] != 0

    #use the mask to locate the values where it is the quarter, and then where the rating is missing
    rows_missing = df.loc[quarter_mask, 'review_scores_checkin'].isna().sum()

    for col in missing_cols:

        #identify the specific quarter and column
        sub_df = df.loc[quarter_mask, col]

        missing_data_table.append(
            {
                'quarter' : quarter,
                'column' : col,
                'missing_count' : sub_df.isna().sum(),
                'min' : sub_df.min(),
                'median' : sub_df.median(),
                'max' : sub_df.max()
            }
        )
#pd.DataFrame(missing_data_table)




#options for replenishing the data
#use the mean if the variance is minial
#build a linear regression using all the other data to predict the ratings



# In[1934]:


#df.columns


# ## Based on the above analysis it makes sense to impute the data using the median values for each calendar time period year

# In[25]:


#create the columns that will hold the missing values and mark them before imputing the median
for col in missing_cols:
    df[f'{col}_missing'] = 0

for col in missing_cols:
    missing_mask = df[col].isna()
    df.loc[missing_mask, f'{col}_missing'] = 1
    print(f"Marked {missing_mask.sum()} missing values in {col}")

#identify the medians for each of the different missing values
for quarter in timeline_cols:
    #create quarter mask to go to that specific quarter
    quarter_mask = df[quarter] != 0
    
    for col in missing_cols:
        #create a missing mask that takes into account quarter mask as well
        miss_mask = quarter_mask & df[col].isna()

        if miss_mask.any(): #if there are any missing masks

            #median for column and quarter
            medians = df.loc[quarter_mask, col].median()

            #this will be added to the dataset to show that the original data was missing
            df.loc[miss_mask, f'{col}_missing'] = 1
            df.loc[miss_mask, col] = medians

#st.write("Missing Data Imputed")


# ## turn all the sparse values into integer or float values

# In[27]:


#check the dtypes and confirm there are no strings
column_list = df.columns.tolist()

#turn the into integer columns
for col in column_list:
    if df[col].dtype not in ('float64', 'int'):
        df[col] = df[col].astype(int).copy()

#for col in column_list:
    #print(df[col].dtype)


# ## remove major outliers

# In[29]:


#create the upper and lower bounds
lower_bound = df['price'].quantile(2.5/100)
upper_bound = df['price'].quantile(97.5/100)

#create a mask
mask = (df['price'] >= lower_bound) & (df['price'] <= upper_bound)
df = df[mask].copy()


# In[31]:


pre_scaled_df = df.copy()


# ## scale non-binary features

# In[33]:


#remove price

columns_to_check = [col for col in df if col != 'price']

#find the min and max for all columns
minis = df[columns_to_check].min()
maxis = df[columns_to_check].max()

#create an empty list for columns to scale
columns_to_scale = []

for col in columns_to_check:

    #find the min and max
    mi = minis[col]
    ma = maxis[col]
    if mi == 0 and ma == 1:
        pass
    else:
        #print(minis[col], maxis[col])
        columns_to_scale.append(col)

#scale the specific columns
scaler = StandardScaler()
df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])

#st.write("Data Scaled")


# In[ ]:





# ## create a function that will run through the different models and once all values are statistically significant return the model information

# In[35]:


#set the random seed
random.seed(18)


#seperate into x and y
x = df.drop('price', axis=1)
y = df['price']

#train test and split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=18)


#look for the logprice
y_train_log = np.log(y_train)
y_test_log = np.log(y_test)


#add in an intercept to design matrices
x_train_int = sm.add_constant(x_train, has_constant='add')
x_test_int = sm.add_constant(x_test, has_constant='add')

#x_train_int.columns.str.contains('const')
#st.write("Train Test Split Created")


# ## Test for Multi Colinearity - Takes to Long to Run on Streamlit so output of code manually entered

# In[428]:


# def vif_calc(x_train_int, exclude_const = True):
#     i = 0
#     #remove the constant
#     if exclude_const == True and 'const' in x_train_int.columns:
#         df_vif = x_train_int.drop('const', axis = 1)
#     else:
#         df_vif = x_train_int.copy()
    
#     #build a dictionary to hold all data
#     vif_data = {}
    
#     #check vif for all columns
#     for i, column in enumerate(df_vif.columns):
#         try:
#             vif = variance_inflation_factor(df_vif.values, i)
#             vif_data[column] = vif
#         except:
#             # Handle any calculation errors
#             vif_data[column] = np.nan

#     return vif_data

# vif_data = vif_calc(x_train_int)
# st.write("VIF calculation completed")


# In[ ]:


## Identify issues and rerun VIF again


# In[483]:


# columns_to_drop = []

# #remove nan values this needs to be done once
# for col, val in vif_data.items():
#     if np.isnan(val):
#         columns_to_drop.append(col)

# #choosing two columns to remove for the base using median price to help with improved pricing outcomes also needs to be done once
# neighbor_cols = [c for c in vif_data.keys() if c.startswith('neighbourhood_')]
# room_cols = [c for c in vif_data.keys() if c.startswith('room_')]
# calendar_cols = [c for c in vif_data.keys() if c.startswith('calendar_')]


# #create a dictionary of medians and their corresponding mean price
# def get_means_for_dummys(df, col_list, dep_var):
#     mean_dict = {}
#     for col in col_list:
        
#         #make sure the dummy variable = 1
#         mask = df[col] == 1
    
#         #get a new df
#         hold_df = df.loc[mask, dep_var]

#         #identfiy the median
#         mean_dict[col] = hold_df.mean()
    
#     return mean_dict


# #find the median column value to remove
# def median_col(mean_dict):
#     sorted_vals = sorted(mean_dict.items(), key= lambda kv:kv[1])
#     index_to_remove = len(sorted_vals) // 2

#     return sorted_vals[index_to_remove][0]


# #remove these columns
# #this couldve been done with a loop but I chose to manually type these out
# neighbor_remove = median_col(get_means_for_dummys(df, neighbor_cols, 'price'))
# room_remove = median_col(get_means_for_dummys(df, room_cols, 'price'))
# calendar_remove = median_col(get_means_for_dummys(df, calendar_cols, 'price'))

# columns_to_drop.append(neighbor_remove)
# columns_to_drop.append(room_remove)
# columns_to_drop.append(calendar_remove)

# st.write("VIF Drop Columns Created")
# print(columns_to_drop)

# x_vif_train = x_train_int.drop(columns=columns_to_drop, errors="ignore").copy()


# ## After Making initial edits to alter the nan and inf numbers run until there is no more multicolinearity

# In[432]:


#x_vif_train = x_train_int.drop(columns=columns_to_drop, errors="ignore").copy()

# i = 1
# while True:
#     # recompute VIFs (make sure your vif_calc drops 'const' internally)
#     vif_dict = vif_calc(x_vif_train, drop_const=True)

#     # if nothing left to evaluate, stop
#     if not vif_dict:
#         st.warning("VIF: no eligible columns left; stopping.")
#         break

#     # handle NaN/inf first (these cause divide-by-zero warnings)
#     bad_cols = [c for c, v in vif_dict.items() if not np.isfinite(v)]
#     if bad_cols:
#         x_vif_train = x_vif_train.drop(columns=bad_cols, errors="ignore")
#         st.write(f"Removed invalid-VIF columns: {bad_cols}")
#         continue

#     # now get the worst offender
#     col_max, vif_max = max(vif_dict.items(), key=lambda kv: kv[1])

#     # threshold to keep
#     if vif_max < 5:
#         st.write(f"All VIFs < 5 after {i-1} iterations.")
#         break

#     # drop it and loop
#     x_vif_train = x_vif_train.drop(columns=[col_max], errors="ignore")
#     st.write(f"Removed {col_max} with VIF {vif_max:.2f} (iteration {i})")
#     i += 1


# In[434]:


# x_vif_train = x_train_int.drop(columns=columns_to_drop).copy()
# #use the vif function to get a dictionary of all vif
# i = 1
# while True:
#     #build the new vif_dict
#     vif_dict = vif_calc(x_vif_train)
    
#     #get the max vif
#     max_vif = max(vif_dict.values())

#     #if the max_vif is less than 5 then say we have a good enough dataset
#     if max_vif < 5:
#         break
    
#     #sort the dictionary so that the max vif is on top
#     sorted_vif_dict = sorted(vif_dict.items(), key = lambda vd: -vd[1])
    
#     #add the column with the largest vif to the columns to drop
#     drop_col = sorted_vif_dict[0][0]

#     #remove the column from the x_vif_train columns
#     x_vif_train = x_vif_train.drop(drop_col, axis = 1)
#     print(f"Removed {drop_col} with a VIF of {max_vif}")
#     st.write(f"VIF Iteration {i} complete!")
#     i += 1


# In[37]:


#x_vif_train.columns

x_vif_train = ['const', 'host_is_superhost', 'bathrooms', 'bedrooms', 'beds',
       'amenities', 'minimum_nights', 'maximum_nights', 'availability_365',
       'number_of_reviews', 'review_scores_rating',
       'review_scores_cleanliness', 'review_scores_checkin',
       'review_scores_communication', 'review_scores_value',
       'instant_bookable', 'calculated_host_listings_count',
       'reviews_per_month', 'host_since_years', 'neighbourhood_Allston',
       'neighbourhood_Back Bay', 'neighbourhood_Bay Village',
       'neighbourhood_Beacon Hill', 'neighbourhood_Brighton',
       'neighbourhood_Charlestown', 'neighbourhood_Chinatown',
       'neighbourhood_Dorchester', 'neighbourhood_Downtown',
       'neighbourhood_East Boston', 'neighbourhood_Fenway',
       'neighbourhood_Hyde Park', 'neighbourhood_Jamaica Plain',
       'neighbourhood_Leather District', 'neighbourhood_Longwood Medical Area',
       'neighbourhood_Mattapan', 'neighbourhood_Mission Hill',
       'neighbourhood_North End', 'neighbourhood_Roslindale',
       'neighbourhood_Roxbury', 'neighbourhood_South Boston Waterfront',
       'neighbourhood_South End', 'neighbourhood_West End',
       'neighbourhood_West Roxbury', 'room_Hotel room', 'room_Private room',
       'room_Shared room', 'calendar_December - 2024', 'calendar_March - 2025',
       'calendar_September - 2024', 'host_response_rate_missing',
       'host_acceptance_rate_missing', 'beds_missing',
       'reviews_per_month_missing']


# In[39]:


x_train_int = x_train_int[x_vif_train]
x_test_int = x_test_int[x_vif_train]


# In[ ]:





# ## Build the Model

# In[69]:


def stepwise_selection(x_train, y_train, threshold = 0.05):
    current_cols = list(x_train.columns)
    
    while True:
        model = sm.OLS(y_train, x_train[current_cols]).fit()

        pvals = model.pvalues.drop('const')
        max_pval_col = pvals.idxmax()
        max_pval = pvals.max()

        if max_pval > threshold:
            print(f"Removed {max_pval_col} pval: {max_pval}")
            current_cols.remove(max_pval_col)
        else:
            break
    return current_cols, model

model_columns, model = stepwise_selection(x_train_int, y_train_log, threshold=0.1)

#st.write("App Passed Phase 7")


# In[1284]:





# ## Test the Model

# In[97]:


#predict using the scaled data
y_pred_train = model.predict(x_train_int[model_columns])
y_pred_test = model.predict(x_test_int[model_columns])

#training model metrics
r2_train = model.rsquared
adj_r2_train = model.rsquared_adj
rmse_train = root_mean_squared_error(y_train_log, y_pred_train)

#testing model metrics
k = len(model.params) - 1
n = len(y_test)

r2_test = r2_score(y_test_log, y_pred_test)
adj_r2_test = 1 - (1-r2_test) *((n-1) / (n - k - 1))
rmse_test = root_mean_squared_error(y_test_log, y_pred_test)


# In[ ]:





# In[ ]:





# # Time to build the App

# ### Introduction

# In[1063]:


#only use the middle so that it is centralized -> little tricks
col1, col2, col3 = st.columns(3)

with col2:
    st.title(":orange[Boston Airbnb Price Model]")
    st.write("This dashboard will analyze 1 year of Boston airbnb data to understand what variables effect the price of airbnb's in the Boston Area.")


# ### General Facts about starting Dataset

# In[1202]:


#median, max, min, total parameters, and total neighborhoods in the model


col1, col2, col3, col4, col5 = st.columns(5)



with col1:
    st.metric('Minimum Price Used in Model', value = "$" +str(pre_scaled_df['price'].min()))

with col2:
    st.metric('Median Price Used in Model', value = "$" + str(pre_scaled_df['price'].median()))

with col3:
    st.metric('Maximum Price Used in Model', value = "$" + str(pre_scaled_df['price'].max()))

with col4:
    st.metric('Total Variables Used in Model', value=len(model.params))

with col5:
    st.metric('Total Boston Neighborhoods used in Model', value = len([col for col in list(model.params.index) if 'neighbourhood' in col]))


# ### Split into Binary and Continuous Variables

# In[1078]:


columns = list(model.params.index)
columns.remove('const')

#create an empty binary and continuous col list
binary_col = []
cont_col = []

#seperate into two seperate lists one for continuous values and one for binary values
for col in columns:
    uni_val = pre_scaled_df[col].nunique()
    if uni_val == 2:
        binary_col.append(col)
    else:
        cont_col.append(col)

#len(binary_col) + len(cont_col)


# In[1122]:


##create the drop down options that are going to be referenced

col1, col2 = st.columns(2)

with col1:
    select_column = st.selectbox("Select a continuous column to view relationship to Airbnb Price", cont_col, key="continuous_column_select")

with col2:
    select_column2 = st.selectbox("Select a binary column to view relationship to Airbnb Price", binary_col, key='binary_column_select')


# ### Variable Effects: Continous

# In[1158]:


#select_column = st.selectbox("Select a continuous column to view relationship to Airbnb Price", cont_col)

#get x and y variables
#x = select_column
#y = 'price'


#create a df to look at the average price
#pre_scaled_df[['price', select_column]].groupby(select_column).mean().reset_index()

#BIG EDITS TO VARIOUS COLUMNS DEPENDING ON HOW MANY UNIQUE COLUMN VALUES THERE ARE
#get the count of unique values
unique_count = pre_scaled_df[select_column].nunique()

#if the count of unique values is low enough than group them as normal without binning
if unique_count <= 10:
    mean_df = pre_scaled_df[['price', select_column]].groupby(select_column).mean().reset_index()

#otherwise binning is necessary
else:
    use_df = pre_scaled_df.copy()
    use_df[f'{select_column}_binned'] = pd.cut(use_df[select_column],
                                              bins=10,
                                              precision=1)

    #get the means by the bin
    mean_df = use_df.groupby(f'{select_column}_binned')['price'].mean().reset_index()

    #create new column bin_label that is a string
    mean_df['bin_label'] = mean_df[f'{select_column}_binned'].astype(str)
    mean_df[select_column] = mean_df[f'{select_column}_binned'].apply(lambda x: round(x.mid))



#create the bar chart
bar = alt.Chart(mean_df).mark_bar(size=64, opacity=1).encode(
    x=alt.X(f"{select_column}:O", 
            title=select_column,
           sort = alt.SortField(field=select_column, order='ascending')),
    y=alt.Y("price:Q", title="Price"),
    tooltip=[alt.Tooltip(f"{select_column}:O"), alt.Tooltip("price:Q", format=",.0f")],
    color=alt.Color(f"{select_column}:Q", scale=alt.Scale(scheme='blues'),legend=None))


#create the trend line
trend = alt.Chart(mean_df).mark_line(color='red', strokeWidth = 3).transform_regression(
    select_column, "price").encode(
    x=alt.X(f"{select_column}:Q"),
    y=alt.Y("price:Q")
    )

#combined the two
combined_chart = (bar + trend).resolve_scale(color='independent').properties(width=650, height=400)

col1, col2 = st.columns(2)

with col1:
    st.subheader(f"{select_column} vs Average Price")
    st.altair_chart(combined_chart, use_container_width=True)


# ### Create a Chart to analyze binary columns

# In[124]:


#create the select column from the binary columns
#select_column2 = st.selectbox("Select a binary column to view relationship to Airbnb Price", binary_col)

pre_scaled_df['legend_col'] = pre_scaled_df[select_column2].map({0: 'No', 1:'Yes'})

#create the violin price chart
bw_plot = alt.Chart(pre_scaled_df).mark_boxplot(size=160, #width of boxplot
                                                box=alt.MarkConfig(stroke='white', strokeWidth=1.5), #outline of the box
                                                median=alt.MarkConfig(color='white', strokeWidth=2), #median
                                                rule=alt.MarkConfig(color='white', strokeWidth=1), #whiskers
                                                ticks=alt.MarkConfig(color='white'), #whisker end ticks
                                                outliers=alt.MarkConfig(color='white', opacity=0.7) #outliers
                                               ).encode(
    x=alt.X(f'legend_col:N', title=select_column2),
    y=alt.Y('price:Q', title='Price'),
color=alt.Color(f'{select_column2}:N', legend=None)
).properties(width=650, height=400)


with col2:
    st.subheader(f'{select_column2} vs Average Price')
    st.altair_chart(bw_plot)


# # Model Output Analysis

# ### Print out the r2 and adj_r2 and rmse for the test and train

# In[122]:


col1, col2, col3, col4, col5, col6 = st.columns(6)

with col1:
    st.metric("R-Squared Train", value= round(r2_train,2))

with col2:
    st.metric("R-Squared Test", value= round(r2_test,2))

with col3:
    st.metric("Adjusted R-Squared Train", value= round(adj_r2_train,2))

with col4:
    st.metric("Adjusted R-Squared Test", value= round(adj_r2_test,2))

with col5:
    st.metric("Root Mean Squared Error Train", value= round(rmse_train,2))

with col6:
    st.metric("Root Mean Squared Error Train", value= round(rmse_test,2))


col1, col2, col3 = st.columns(3)

with col1:
    st.write("The R-Squared measure explains the variance in the price the model explains. This means the model explains ~65% of the variation.")

with col2:
    st.write("The Adjusted R-Squared penalizes for the number of predictors(columns) used in the model. These results show similar information as the R-Squared.")

with col3:
    st.write("The Root Mean Squared Error Test shows the average error size from the price. This results show that predictions errors across splits are consistent and usually around ~42.5%. This means the predicted values are normally within ~42.5% of the actual price.")



# ## Variables and there effects

# In[266]:


#overall effects
effect_df = model.params.to_frame("coef").reset_index().rename(columns = {"index": 'Variable Name'})

#create a dataframe that holds the varname and coef
effect_df = effect_df.sort_values(by="coef")

#add back in rows that were removed to make sure we could accurately calculate VIF
new_rows = [
    {'Variable Name': 'neighbourhood_South Boston', 'coef' : 0},
    {'Variable Name': 'room_Entire home/apt', 'coef' : 0},
    {'Variable Name': 'calendar_June - 2025', 'coef' : 0}
]

effect_df = pd.concat([effect_df, pd.DataFrame(new_rows)], ignore_index=True)

effect_df['Percent Effect'] = effect_df['coef'].apply(lambda x: (np.exp(x) - 1) * 100)

#split into the 4 different dataframes that will be used to pick apart the dataframe and create the visuals
#lists the prefixes used and const as they will be used to get an "others" dataframe
list_of_prefixes = ('neighbourhood', 'room', 'calendar', 'const')

neighbourhood_df = effect_df[effect_df['Variable Name'].str.startswith('neighbourhood')]
room_list = effect_df[effect_df['Variable Name'].str.startswith('room')]
calendar_list = effect_df[effect_df['Variable Name'].str.startswith('calendar')]
others = effect_df[~effect_df['Variable Name'].str.startswith(list_of_prefixes)]

#create a function to create the different bar charts that will be used
def create_bar(df, x, y, colors = 'blues'):
    bar = alt.Chart(df).mark_bar(size=64, opacity=1).encode(
     x=alt.X(f"{x}:O", 
             title=x,
            sort = alt.SortField(field=y, order='ascending')),
     y=alt.Y(f"{y}:Q", title=y),
     tooltip=[alt.Tooltip(f"{x}:O"), alt.Tooltip(f"{y}:Q", format=",.0f")],
     color=alt.Color(f"{y}:Q", scale=alt.Scale(scheme=colors),legend=None)).properties(width=650, height=400)

    return bar

neighbourhood_chart = create_bar(neighbourhood_df, 'Variable Name', 'Percent Effect')


st.subheader(f'Neighbourhood vs Percent Effect on Price')
st.altair_chart(neighbourhood_chart)
    


col1, col2 = st.columns(2)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[120]:


st.title(":orange[Conclusion]")
st.write("Bottom line: this model is doing great with the data we’ve got. It explains about two-thirds of price variation (R² 0.645 train / 0.659 test), and even after penalizing for feature count the story holds (adj-R² 0.643 / 0.653), so we’re capturing real signal—not just fitting noise. Errors on the log scale are steady at 0.384 (train) and 0.366 (test), roughly a ~47% / ~44% typical gap, which is solid given we’re mostly using host-provided listing details and not richer property data (condition, square footage, comps, events). The test slightly edging the train = nice generalization. Overall, given the available data, this model reliably captures how listing features affect nightly price in Boston and does it with stable, well-generalized, and consistent performance!")
st.write('Link to Data Source: https://insideairbnb.com/get-the-data/')

