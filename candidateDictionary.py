import pandas as pd
import json

# Read the CSV files into pandas dataframes
first_csv_path = "committee_summary_2024.csv"
second_csv_path = "candidate_summary_2024.csv"

# Assuming the first CSV file has a column named 'CAND_ID'
# and the second CSV file has a column named 'cand_id'
first_df = pd.read_csv(first_csv_path)
second_df = pd.read_csv(second_csv_path)

# Merge dataframes based on the 'CAND_ID' column
merged_df = pd.merge(first_df, second_df, how='inner', left_on='CAND_ID', right_on='Cand_Id')

# Create a JSON file to store the result
json_file_path = "committee_candidate.json"

# Create a list to store the JSON records
json_records = []

# Iterate through the merged dataframe and create JSON records
for _, row in merged_df.iterrows():
    record = {
        'committee_name': row['CMTE_NM'],
        'committee_id': row['CMTE_ID'],
        'candidate_name': row['Cand_Name'],
        'candidate_id': row['Cand_Id'],
        'party_affiliation': row['Cand_Party_Affiliation']
    }
    json_records.append(record)

# Write the JSON records to a file
with open(json_file_path, 'w') as json_file:
    json.dump(json_records, json_file, indent=2)

print(f"JSON file created at: {json_file_path}")
