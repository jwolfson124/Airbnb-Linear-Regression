#!/usr/bin/env python
# coding: utf-8

# In[163]:


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

# In[50]:


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
st.write("App Passed Phase 1")


# In[173]:


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


# In[5]:





# ## Identify columns that are not consistent and remove them from df

# In[175]:


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


# In[679]:





# # Identify Columns that will not be useful to the algorythm

# In[177]:


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

st.write('App Passed Phase 2')


# ## Change any datetime columns to integer values

# In[179]:


#columns that need to be changed
print(small_df.select_dtypes(include=['datetime', 'datetime64[ns]', 'datetimetz']).columns)

#change host since to get total days as a host, also remove the NAN values, and drop the old column
small_df = small_df[~small_df['host_since'].isna()]
small_df['host since'] = (datetime.now() - small_df['host_since']).dt.days.astype(int)
small_df = small_df.drop(columns='host_since')


#turn the small_df calendar last scraped into the month and the year as a string
small_df['calendar_last_scraped'] = small_df['calendar_last_scraped'].dt.strftime('%B - %Y')



# In[96]:





# ## Change categorical values into dummy variables

# In[181]:


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

# In[183]:


#the ast.literal_eval turns the string that holds a list into just a list of the different amenities
#the lambda(x: ','.join(x)) is going to then take the list of strings and turn them into one long list
#this will allow the string to be turned into dummies using the "," seperator
df['amenities'] = df['amenities'].apply(ast.literal_eval).apply(lambda x: ','.join(x))

#df_dummies = df['amenities'].str.get_dummies(sep=',')
#df_dummies


# In[185]:


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

st.write("App Passed Phase 3")


# ## edit the data so that it will show entire vs shared vs private room as opposed to all the different options

# In[2208]:


#df['room_type']


# ## Get dummy values and apply prefix to help with organizatioon

# In[187]:


#create a list of dummy columns
dummy_cols = df.select_dtypes(include=['object']).columns

#identify what the prefix for each of the columns will be
pattern = re.compile(r'^[^_]+')
prefix = [pattern.match(c).group(0) for c in dummy_cols]

#get the dummys and apply the prefixs
dummy_values = pd.get_dummies(df[dummy_cols],prefix=prefix, dtype='uint8', sparse=True)

df = pd.concat([df, dummy_values], axis=1).drop(columns=dummy_cols)


# In[189]:


timeline_cols = df.columns[df.columns.str.contains('calendar')]
#timeline_cols


# In[ ]:





# ## identify missing data and how to deal with it the means, medians, max, and min to understand how similar the information is

# In[191]:


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

# In[193]:


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

st.write("App Passed Phase 4")


# ## turn all the sparse values into integer or float values

# In[195]:


#check the dtypes and confirm there are no strings
column_list = df.columns.tolist()

#turn the into integer columns
for col in column_list:
    if df[col].dtype not in ('float64', 'int'):
        df[col] = df[col].astype(int).copy()

#for col in column_list:
    #print(df[col].dtype)


# ## remove major outliers

# In[197]:


#create the upper and lower bounds
lower_bound = df['price'].quantile(2.5/100)
upper_bound = df['price'].quantile(97.5/100)

#create a mask
mask = (df['price'] >= lower_bound) & (df['price'] <= upper_bound)
df = df[mask].copy()


# ## scale non-binary features

# In[199]:


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

st.write("App Passed Phase 5")


# In[ ]:





# ## create a function that will run through the different models and once all values are statistically significant return the model information

# In[205]:


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


# ## Test for Multi Colinearity

# In[207]:


def vif_calc(x_train_int, exclude_const = True):
    
    #remove the constant
    if exclude_const == True and 'const' in x_train_int.columns:
        df_vif = x_train_int.drop('const', axis = 1)
    else:
        df_vif = x_train_int.copy()
    
    #build a dictionary to hold all data
    vif_data = {}
    
    #check vif for all columns
    for i, column in enumerate(df_vif.columns):
        try:
            vif = variance_inflation_factor(df_vif.values, i)
            vif_data[column] = vif
        except:
            # Handle any calculation errors
            vif_data[column] = np.nan

    return vif_data

vif_data = vif_calc(x_train_int)
st.write("VIF calculation completed")


# In[ ]:


## Identify issues and rerun VIF again


# In[208]:


columns_to_drop = []

#remove nan values this needs to be done once
for col, val in vif_data.items():
    if np.isnan(val):
        columns_to_drop.append(col)

#choosing two columns to remove for the base using median price to help with improved pricing outcomes also needs to be done once
neighbor_cols = [c for c in vif_data.keys() if c.startswith('neighbourhood_')]
room_cols = [c for c in vif_data.keys() if c.startswith('room_')]
calendar_cols = [c for c in vif_data.keys() if c.startswith('calendar_')]


#create a dictionary of medians and their corresponding mean price
def get_means_for_dummys(df, col_list, dep_var):
    mean_dict = {}
    for col in col_list:
        
        #make sure the dummy variable = 1
        mask = df[col] == 1
    
        #get a new df
        hold_df = df.loc[mask, dep_var]

        #identfiy the median
        mean_dict[col] = hold_df.mean()
    
    return mean_dict


#find the median column value to remove
def median_col(mean_dict):
    sorted_vals = sorted(mean_dict.items(), key= lambda kv:kv[1])
    index_to_remove = len(sorted_vals) // 2

    return sorted_vals[index_to_remove][0]


#remove these columns
#this couldve been done with a loop but I chose to manually type these out
neighbor_remove = median_col(get_means_for_dummys(df, neighbor_cols, 'price'))
room_remove = median_col(get_means_for_dummys(df, room_cols, 'price'))
calendar_remove = median_col(get_means_for_dummys(df, calendar_cols, 'price'))

columns_to_drop.append(neighbor_remove)
columns_to_drop.append(room_remove)
columns_to_drop.append(calendar_remove)

st.write("App Passed Phase 6")
print(columns_to_drop)


# ## After Making initial edits to alter the nan and inf numbers run until there is no more multicolinearity

# In[213]:


x_vif_train = x_train_int.drop(columns=columns_to_drop).copy()
#use the vif function to get a dictionary of all vif

while True:
    #build the new vif_dict
    vif_dict = vif_calc(x_vif_train)
    
    #get the max vif
    max_vif = max(vif_dict.values())

    #if the max_vif is less than 5 then say we have a good enough dataset
    if max_vif < 5:
        break
    
    #sort the dictionary so that the max vif is on top
    sorted_vif_dict = sorted(vif_dict.items(), key = lambda vd: -vd[1])
    
    #add the column with the largest vif to the columns to drop
    drop_col = sorted_vif_dict[0][0]

    #remove the column from the x_vif_train columns
    x_vif_train = x_vif_train.drop(drop_col, axis = 1)
    print(f"Removed {drop_col} with a VIF of {max_vif}")



# In[219]:


x_train_int = x_train_int[x_vif_train.columns]
x_test_int = x_test_int[x_vif_train.columns]


# In[ ]:





# ## Build the Model

# In[221]:


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

st.write("App Passed Phase 7")


# ## Test the Model

# In[223]:


#predict based on the model
y_pred_log = model.predict(x_test_int[model_columns])
y_pred = np.exp(y_pred_log)

r2 = r2_score(y_test, y_pred)
print("R-Squared of the Test Data:", r2)

mse = root_mean_squared_error(y_test, y_pred)
print("Mean Squared Error of the Test Data:", mse)


st.write("App Passed Phase 8")


# In[2026]:





# In[ ]:





# In[ ]:





# ## Create and run the Linear Model Without Any Data Manipulation

# In[ ]:





# ## Edits to the model due to outputs

# In[ ]:





# ## Remove all outputs that have the same values

# In[ ]:




