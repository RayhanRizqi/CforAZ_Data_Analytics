import pandas as pd

df = pd.read_csv("CforAZ_Data_Analytics\CFAZ Modeling Data.csv")

#get rid of useless irrelevant weak ugly column "VANID"
df.drop(columns=['VANID'], inplace=True)
df.drop(columns=['Employer'], inplace=True)
df.drop(columns=['Occupation'], inplace=True)
df.drop(columns=['City'], inplace=True)
df.drop(columns=['State'], inplace=True)

# Adjust the process_string_corrected function to specifically extract Gender, Ethnicity, and Networth
def extract_bio_details(bio_string):
    # Split the bio string by four consecutive spaces
    split_by_spaces = bio_string.split("    ")
    
    # Extract Gender and Ethnicity from the first part, and attempt to find a networth value in the second part
    gender_ethnicity = split_by_spaces[0].split(",") if len(split_by_spaces) > 0 else ["", ""]
    networth_string = split_by_spaces[1][15:] if len(split_by_spaces) > 1 else ""
    networth_value = ''.join(filter(str.isdigit, networth_string))
    networth = int(networth_value) if networth_value else None
    
    # Prepare the output to match Gender, Ethnicity, and Networth
    gender = gender_ethnicity[0] if len(gender_ethnicity) > 0 else None
    ethnicity = gender_ethnicity[1] if len(gender_ethnicity) > 1 else None
    
    return gender, ethnicity, networth

# Apply the extract_bio_details function to the "Bio" column and create new columns
df[['Gender', 'Ethnicity', 'Networth']] = df.apply(lambda row: pd.Series(extract_bio_details(row['Bio'])), axis=1)
df.drop(columns=['Bio'], inplace=True)

# Display the modified DataFrame columns to verify the new structure
print(df.columns)

# Apply get_dummies to "Gender" and "Ethnicity" columns to create one-hot encoded columns
gender_dummies = pd.get_dummies(df['Gender'], prefix='Gender')
ethnicity_dummies = pd.get_dummies(df['Ethnicity'], prefix='Ethnicity')

# Concatenate these new one-hot encoded columns with the original DataFrame
df_with_dummies = pd.concat([df, gender_dummies], axis=1)
df_with_dummies = pd.concat([df_with_dummies, ethnicity_dummies], axis=1)
# Display the modified DataFrame columns to verify the new structure with dummies
print(df_with_dummies.columns)
