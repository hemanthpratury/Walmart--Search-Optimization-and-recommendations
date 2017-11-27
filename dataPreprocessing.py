
import pandas as pd
import numpy as np

if __name__ == '__main__':
    ############# READ RAW DATA FROM FILE ##############
    input_folder_path = r"C:\Users\Hemanth"

    filename1 = "\dataset2.txt"
    filename2 = "\dataset2.txt"

    header_names = ["term", "product_id", "language", "product_impressions", "Product_clicks", "cart_adds",
                    "cart_start", "checkout", "order"]
    df_dly_hdp1 = pd.read_csv(input_folder_path + filename, sep='\t', names=header_names)
    df_dly_hdp2 = pd.read_csv(input_folder_path + filename, sep='\t', names=header_names)

    frames = [df_dly_hdp1, df_dly_hdp2]
    result = pd.concat(frames, ignore_index=True)

    # filter out French language records
    result = (result[result['language'] == 'English'])

    joined_df_dly = (
        result[['term', 'product_id', 'product_impressions', 'Product_clicks', 'cart_adds', 'checkout', 'order']])


# print(df_dly_hdp)


joined_df_dly['searchTerm'] = joined_df_dly['term']
joined_df_dly['ATR'] = joined_df_dly['cart_adds'] / joined_df_dly['product_impressions']
joined_df_dly['CTR'] = joined_df_dly['Product_clicks'] / joined_df_dly['product_impressions']
joined_df_dly['conv'] = joined_df_dly['order'] / joined_df_dly['product_impressions']

columns_all = ['searchTerm', 'product_id', 'product_impressions', 'Product_clicks', 'cart_adds', 'order', 'CTR', 'ATR',
               'conv']

# columns for input

joined_df_dly = joined_df_dly[list(columns_all)].reset_index()

print(joined_df_dly.info())

###########################################3

# print(joined_df_dly.tail(50))

# Concat null results and positive results - Final step

# columns for input

final_merged_dataset = joined_df_dly[list(columns_all_final)]
final_merged_dataset = final_merged_dataset.replace(np.nan, 0)
final_merged_dataset = final_merged_dataset.drop_duplicates()

final_merged_dataset.to_csv('C:\Users\Hemanth\final_merged_two_dataset.csv', sep=',',
                            index=False)

# result = df_dly_null.append(df_dly_pos, ignore_index=True)
print(final_merged_dataset.head())



