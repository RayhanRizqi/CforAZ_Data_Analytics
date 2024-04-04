import pandas as pd
import re
from datetime import datetime

df = pd.read_csv("CforAZ_Data_Analytics\CFAZ Modeling Data.csv")

# get rid of useless irrelevant weak ugly column "VANID"
df.drop(columns=['VANID'], inplace=True)
df.drop(columns=['Employer'], inplace=True)
df.drop(columns=['Occupation'], inplace=True)
df.drop(columns=['City'], inplace=True)
df.drop(columns=['State'], inplace=True)
df.drop(columns=['Giving History'], inplace=True)

# add columns about previous gives
df['Max Give'] = 0
df['Months Since Last Donation'] = 0
df['Most Recent Give Amount'] = 0
df['Number of Gives'] = 0

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
# print(df.columns)

# Apply get_dummies to "Gender" and "Ethnicity" columns to create one-hot encoded columns
gender_dummies = pd.get_dummies(df['Gender'], prefix='Gender')
ethnicity_dummies = pd.get_dummies(df['Ethnicity'], prefix='Ethnicity')

# Concatenate these new one-hot encoded columns with the original DataFrame
df_with_dummies = pd.concat([df, gender_dummies], axis=1)
df_with_dummies = pd.concat([df_with_dummies, ethnicity_dummies], axis=1)

# Prettiy Gender columns
for col in df_with_dummies.columns:
    if "Gender_" in col:
        newCol = col.split("_")
        df_with_dummies = df_with_dummies.rename(columns={col: newCol[1]})

# Prettify Ethnicity columns
for col in df_with_dummies.columns:
    if "Ethnicity_" in col:
        newCol = col.split(" ")
        df_with_dummies = df_with_dummies.rename(columns={col: newCol[2]})


ethnicColumns = ["afam", "chinese", "korean", "indian", "iranian", "japanese", "muslim", "white", "greek", "italian", "irish", "jewish", "latinx"]

# very questionable variable name, might as well call it ethnic Cleansing table
ethnicConversions = {"afam": "isAfam", "chinese": "isAsian", "korean": "isAsian",
                     "indian": "isAsian", "iranian": "isAsian", "japanese": "isAsian",
                     "muslim": "isAsian", "white": "isWhite", "greek": "isWhite",
                     "italian": "isWhite", "irish": "isIrish", "jewish": "isJewish",
                     "latinx": "isLatinx"}

# for col in df_with_dummies.columns:
#     if col in ethnicColumns:
#         print(col + ": " + str(df_with_dummies[col].sum()))
        
#         df_with_dummies = df_with_dummies.
#         ethnicConversions[col]

# converting ethnicity columns to grouped ethnicity
for original, new in ethnicConversions.items():
    if new not in df_with_dummies.columns:
        df_with_dummies[new] = 0
    df_with_dummies[new] = df_with_dummies.apply(lambda row: 1 if row[original] == 1 or row[new] == 1 else 0, axis=1)
    df_with_dummies = df_with_dummies.drop(columns={original})

# handling multi ethnic people
for col in df_with_dummies:
    if "/" in col:
        for eth in col.split("/"):
            df_with_dummies[ethnicConversions[eth]] = 1
        df_with_dummies = df_with_dummies.drop(columns={col})

# Getting rid of initial Gender and Ethnicity columns
df_with_dummies.drop(columns=['Gender'], inplace=True)
df_with_dummies.drop(columns=['Ethnicity'], inplace=True)

# Display the modified DataFrame columns to verify the new structure with dummies
# print(df_with_dummies.columns)
# print(df_with_dummies.head())

# 
# print(df_with_dummies["Bio Contributions"])
count = 0

for i in range(len(df_with_dummies)):
    wordString = df_with_dummies.iloc[i]["Bio Contributions"].split("   ")
    
    # Reset Max Give
    maxGive = 0

    # Reset months_since_donation
    months_since_donation = -1
    
    # handles if the row start with 
    if (wordString[0] == "--------- all_contributions:"):
        # removes the prefix "---- all contributions:"
        donations = wordString[1:]
        
        # gets rid of leading & trailing white spaces & removes empty strings form lists, cuz for some reason they have empty strings
        donations = [item for item in [item.strip() for item in donations] if item]

        

        # get the list of all the numbers in the middle of the donation, then get the max of all of them
        maxGive = int(max([float(s.split(' - ')[1]) for s in donations]))

        # set the max Give in the df coluns "Max Give"
        df_with_dummies.at[i, "Max Give"] = maxGive


        # Get dates for every donation
        donation_dates = [datetime.strptime(date.split(' - ')[0], "%Y-%m-%d %H:%M:%S") for date in donations]

        # Most recent date
        most_recent_dono = max(donation_dates)

        # Current Date
        current_date = datetime.now()

        # How long its been since last donation (in months)
        months_since_donation = (current_date.year - most_recent_dono.year) * 12 + (current_date.month - most_recent_dono.month)

        df_with_dummies.at[i, "Months Since Last Donation"] = months_since_donation
    else:
        sections = ''.join(wordString)

        # Split the sections by the "---------" delimiter
        sections = [item for item in [item.strip() for item in sections.split("---------")] if item]
        
        # section = "Most recent ...", "Presidential", "Issue", "Most recent House"
        for section in sections:
            # Dont include the presidential donations
            if "Presidential" in section:
                continue

            # Gets rid of the "Most recent ..." header and splits into list of donations in the section
            section = [item for item in ''.join(section.split("\n")[1:]).split(", ") if item]
            section = [item for item in section if item != ","]

            # gets max dono amount
            for donation in section:
                # splits by empty strings and ","
                pattern = r'[ ,]+'
                donation = re.split(pattern, donation)

                #removes empty strings
                donation = [item for item in donation if item]

                # if it includes donation amount, get the donation amount
                if len(donation) == 3:
                    # Sometimes in place of the donation amount, its replaced by a uppercase character, so this takes it into account
                    try:
                        maxGive = max(maxGive, int(donation[len(donation)-1]))
                    except ValueError:
                        continue
                
                # month abbreviation to number
                month_to_num = {
                    'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
                    'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
                }

                # month and year of the donation
                month_abbr = donation[0][:3]
                year = int(donation[0][3:])

                # convert the month string to int
                month_num = month_to_num[month_abbr.lower()]

                # date object
                date_object = datetime(year, month_num, 1)

            # Keep track of most recent donation onject
            if (most_recent_dono == -1):
                most_recent_dono = date_object
            # check if the date is most recent
            most_recent_dono = max(date_object, most_recent_dono)

        # now that we have the max dono value, set it in the dataframe at column "Max Give"
        df_with_dummies.at[i, "Max Give"] = maxGive

        # current date
        current_date = datetime.now()

        # How long its been since last donation (in months)
        months_since_donation = (current_date.year - most_recent_dono.year) * 12 + (current_date.month - most_recent_dono.month)

        # set most_recent_dono into the dataframe
        df_with_dummies.at[i, "Months Since Last Donation"] = months_since_donation