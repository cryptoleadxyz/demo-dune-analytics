# Author: CryptoLead at https://www.cryptolead.xyz
# Date: 2023-03-19
# Message: Hey there fellow coders! If you found my code helpful and want to show your support, consider buying me a coffee or two. Your donations help me keep improving the code and creating more awesome stuff for the community. Thanks for your support!
# Donation: cryptolead.eth or 0xa2c35DA418f52ed89Ba18d51DbA314EB1dc396d0

import os
import matplotlib.pyplot as plt
from datetime import date
from dotenv import load_dotenv
from duneanalytics import DuneAnalytics
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import warnings

warnings.filterwarnings("ignore")


def pandas_output_setting():
    """Set pandas _output display setting"""
    pd.set_option("display.max_rows", 500)
    pd.set_option("display.max_columns", None)
    ##pd.set_option('display.max_columns', 500)
    pd.set_option("display.width", 140)
    pd.set_option("display.max_colwidth", None)
    pd.options.mode.chained_assignment = None  # default='warn'


# pandas_output_setting()


# Function that extract full data and column from DuneSQL query result
def extract_col_and_data_from_dunesql_result(query_id):
    result_id = dune.query_result_id_v3(query_id=query_id)
    full_data = dune.get_execution_result(result_id)
    column = full_data["data"]["get_execution"]["execution_succeeded"]["columns"]
    data = full_data["data"]["get_execution"]["execution_succeeded"]["data"]
    return (column, data)


def create_df_from_dune_output_column_and_data(dune_column, dune_result_data):
    # Set up empty df with columns only
    df = pd.DataFrame(columns=dune_column)
    # Iterate through the row-level entry, and concatenate (or append) into the existing df
    for row in dune_result_data:
        new_row = pd.DataFrame.from_records(row, index=[0])
        df = pd.concat([df, new_row], ignore_index=True)
    # Result final df
    return df


# Connect to user's Dune account
load_dotenv()
dune = DuneAnalytics(os.getenv("DUNE_USERNAME"), os.getenv("DUNE_PASSWORD"))
dune.login()
dune.fetch_auth_token()

# Extract data specifically from query_id={specific query id}
# Get Ethereum block data, and turn it into df
eth_block_column, eth_block_actual_data = extract_col_and_data_from_dunesql_result(
    1946225
)
df_eth_block = create_df_from_dune_output_column_and_data(
    eth_block_column, eth_block_actual_data
)
# Remove rows with duplicated block number, and then do an "insert" check
block_num_list = df_eth_block["number"].tolist()
assert len(block_num_list) == len(list(set(block_num_list)))

# Get Ethereum transaction data, and turn it into df
eth_txn_column, eth_txn_actual_data = extract_col_and_data_from_dunesql_result(2213440)
# Get rid of "access_list" from both the column list and data since we don't need it
eth_txn_column.remove("access_list")
for row in eth_txn_actual_data:
    del row["access_list"]
df_eth_txn = create_df_from_dune_output_column_and_data(
    eth_txn_column, eth_txn_actual_data
)

print(df_eth_block)
print(df_eth_txn)


# Aggregate `df_eth_txn` to get 1) number of transactions per block, 2) summed "gas_used", and 3) max "gas_used"
df_eth_txn_agg = df_eth_txn.groupby(by=["block_number"]).agg(
    {"gas_used": ["sum", "max"], "block_hash": ["count"]}
)
# Flatten the multilevel index of `df_eth_txn_agg`
df_eth_txn_agg.columns = ["_".join(col) for col in df_eth_txn_agg.columns]
df_eth_txn_agg = df_eth_txn_agg.reset_index()
df_eth_txn_agg.rename(columns={"block_hash_count": "txn_count"}, inplace=True)

# Then merge `df_eth_txn_agg` with `df_eth_block` by "block_number"
df_final = df_eth_block.merge(
    df_eth_txn_agg, how="left", left_on="number", right_on="block_number"
)
# Drop rows that don't have a valid value on "txn_count" column
df_final["txn_count"].replace("", np.nan, inplace=True)
df_final.dropna(subset=["txn_count"], inplace=True)
df_final = df_final[
    1:
]  # drop the first row since we aren't sure summed and max of "gas_used" include all data due to truncated data used
# Sort by time
df_final["time"] = pd.to_datetime(df_final["time"])
df_final = df_final.sort_values(by="time", kind="mergesort", ascending=True)
# Set "time" column as index
df_final.set_index("time", inplace=True)

# Correlation between two variables
corr, _ = pearsonr(df_final["gas_used_sum"], df_final["txn_count"])
print(
    f"The Pearson correlation between variables gas_used_sum and txn_count is {corr}."
)

# Visualization
var1_color = "brown"
var2_color = "blue"
plt.rcParams["figure.figsize"] = [7.50, 3.50]
plt.rcParams["figure.autolayout"] = True
ax1 = df_final["gas_used_sum"].plot(color=var1_color, label="Total gas used")
ax2 = df_final["txn_count"].plot(
    color=var2_color, secondary_y=True, label="Number of transactions"
)
h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
# Set title and axis labels
ax1.set_title("Time-series of gas used and number of transactions per block")
ax1.set_xlabel(f"Time on {date.today()}")
ax1.set_ylabel("Total gas used per block", color=var1_color)
ax2.set_ylabel("Total number of transactions per block", color=var2_color)
plt.legend(h1 + h2, l1 + l2, loc=2)
plt.show()
