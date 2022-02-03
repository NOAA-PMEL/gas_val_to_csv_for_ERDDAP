from parse_all_into_csv7_dry_Tcorr_10XX import *
import pandas as pd
import glob

list_of_SN = ['1004','1005','1006','1008','1009']

list_of_path_to_data = ['./data/1004/20210512/','./data/1005/20210514/',\
    './data/1006/20210430/','./data/1008/20210429/','./data/1009/20210428/']

validation_filenames = ['./data/1004/20210512/1004_Validation_20210512-210237.txt',\
    './data/1005/20210514/1005_Validation_20210514-004141.txt',\
    './data/1006/20210430/1006_Validation_20210430-combo.txt',\
    './data/1008/20210429/1008_Validation_20210429-combo.txt',\
    './data/1009/20210428/1009_Validation_20210427-171323.txt']

for idx_sn, this_sn in enumerate(list_of_SN):
    #### Good stuff ####
    #path_to_data='./data/' + this_sn + '/'
    path_to_data = list_of_path_to_data[idx_sn]
    path_to_ALL= path_to_data + 'ALL'
    filenames=glob.glob(path_to_ALL + '/2021*.txt')
    filenames.sort()
    list_of_df = []
    list_of_dry_df = []
    list_of_dry_df_sync = []
    list_of_stats_df = []
    list_of_flags_df = []
    list_of_coeff_sync_df =[]
    for idx, filename in enumerate(filenames):
        print(filename)
        # bypass files without coefficients
        COEFF_found_in_file=False
        with open(filename) as f:
            if 'COEFF' in f.read():
                COEFF_found_in_file =True
        f.close()
        del f
        if ( COEFF_found_in_file ):
            df_file = parse_all_file_df_w_summary_v2(filename)
            df_file = all_df_final_reorder_10XX(df_file)
            df_dry_sync, df_dry = loadASVall_dry(filename)
            df_stats = loadASVall_stats(filename)
            df_flags = loadASVall_flags(filename)
            df_ts_flags = add_ts_to_flags(df_flags,df_stats)
            df_coeff_sync = loadASVall_coeff_sync(filename)
            list_of_df.append(df_file)
            list_of_dry_df.append(df_dry)
            list_of_stats_df.append(df_stats)
            list_of_flags_df.append(df_ts_flags)
            #print(f'percent done = {100*idx/len(filenames)}%')
            list_of_dry_df_sync.append(df_dry_sync)
            list_of_coeff_sync_df.append(df_coeff_sync)

    super_big_df = pd.concat(list_of_df,axis=0,ignore_index=True)
    super_big_dry_df = pd.concat(list_of_dry_df,axis=0,ignore_index=True)
    super_big_dry_df_sync = pd.concat(list_of_dry_df_sync,axis=0,ignore_index=True)
    super_big_stats_df = pd.concat(list_of_stats_df,axis=0,ignore_index=True)
    super_big_flags_df = pd.concat(list_of_flags_df,axis=0,ignore_index=True)
    super_big_coeff_sync_df = pd.concat(list_of_coeff_sync_df,axis=0,ignore_index=True)
    super_big_df.to_csv(path_to_data + 'raw_w_summary_' + this_sn + '.csv', index=False)

    super_big_stats_df.to_csv(path_to_data + 'stats_' + this_sn + '.csv', index=False)

    validation_filename = validation_filenames[idx_sn]

    super_big_val_df = load_Val_file(validation_filename,super_big_dry_df_sync,\
        super_big_stats_df,super_big_flags_df,super_big_coeff_sync_df)
    super_big_val_df.to_csv(path_to_data + 'Report_Summary_parsed_from_every_txt_file_'+ this_sn + '.csv',index=False)
    print(f'#### {this_sn} is done ####')