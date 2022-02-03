import pandas as pd
import json


df_10XX = pd.read_csv('./data/1004/20210512/raw_w_summary_1004.csv')
df_Saildrone = pd.read_csv('./data/3CA8A2533/raw_w_summary_3CA8A2533.csv')

print(df_10XX.head())
print(df_Saildrone.head())

# Check the data for column compatibility
pd.set_option('max_columns',None)
df_10XX_cols = list(df_10XX.columns.values)
df_Saildrone_cols = list(df_Saildrone.columns.values)
pd.reset_option('max_columns')

print (f'num cols in 10XX = {len(df_10XX_cols)}')
print (f'num cols in Saildrone = {len(df_Saildrone_cols)}')

found_in_10XX_but_not_found_in_Saildrone = []
for idx_10XX, this_10XX_col in enumerate(df_10XX_cols):
    if ( this_10XX_col not in df_Saildrone_cols):
        found_in_10XX_but_not_found_in_Saildrone.append(this_10XX_col)

found_in_Saildrone_but_not_found_in_10XX = []
for idx_Saildrone, this_Saildrone_col in enumerate(df_Saildrone_cols):
    if ( this_Saildrone_col not in df_10XX_cols):
        found_in_Saildrone_but_not_found_in_10XX.append(this_Saildrone_col)

if ( len(found_in_10XX_but_not_found_in_Saildrone) != 0 ):
    print(f'extra columns in 10XX {found_in_10XX_but_not_found_in_Saildrone}')
else:
    print('Everything that was in 10XX is found in Saildrone')

if ( len(found_in_Saildrone_but_not_found_in_10XX) != 0 ):
    print(f'extra columns in Saildrone {found_in_Saildrone_but_not_found_in_10XX}')
else:
    print('Everything that was in Saildrone is found in 10XX')

#### Check order of columns ####
out_of_order_10XX = []
for idx_10XX, this_10XX_col in enumerate(df_10XX_cols):
    if ( this_10XX_col != df_Saildrone_cols[idx_10XX] ):
        out_of_order_10XX.append(this_10XX_col)

print(f'The following 10XX columns are out of order {out_of_order_10XX}')

print("#### Raw df columns SHOULD BE ####")
print(df_Saildrone_cols)
f = open('./Saildrone_raw_df_columns.json','w',encoding='utf-8')
json.dump(df_Saildrone_cols, f, indent=4, ensure_ascii=False, allow_nan=True)
f.close()