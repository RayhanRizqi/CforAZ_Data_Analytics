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
df['Average Give Amount'] = 0

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
    
    # handles if the row start with "--------- all_contributions:"
    if (wordString[0] == "--------- all_contributions:"):
        # removes the prefix "---- all contributions:"
        donations = wordString[1:]
        
        # gets rid of leading & trailing white spaces & removes empty strings form lists, cuz for some reason they have empty strings
        donations = [item for item in [item.strip() for item in donations] if item]

        

        # get the list of all the numbers in the middle of the donation, then get the max of all of them
        maxGive = int(max([float(s.split(' - ')[1]) for s in donations]))

        # set the max Give in the df coluns "Max Give"
        df_with_dummies.at[i, "Max Give"] = maxGive

        donation_amounts = [float(s.split(' - ')[1]) for s in donations]

        # Set average give in df at column "Average Give Amount"
        df_with_dummies.at[i, "Average Give Amount"] = sum(donation_amounts) / len(donation_amounts)


        # get the amount of the latest give
        most_recent_dono_amount = int(float(donations[len(donations)-1].split(' - ')[1]))

        # set most recent dono amount in "Most Recent Give Amount"
        df_with_dummies.at[i, "Most Recent Give Amount"] = most_recent_dono_amount

        # Get dates for every donation
        donation_dates = [datetime.strptime(date.split(' - ')[0], "%Y-%m-%d %H:%M:%S") for date in donations]

        # Most recent date
        most_recent_dono = max(donation_dates)

        # Current Date
        current_date = datetime.now()

        # How long its been since last donation (in months)
        months_since_donation = (current_date.year - most_recent_dono.year) * 12 + (current_date.month - most_recent_dono.month)

        df_with_dummies.at[i, "Months Since Last Donation"] = months_since_donation
        # print(donations)

        # setting the amount of donations in df under column "Number of Gives"
        df_with_dummies.at[i, "Number of Gives"] = len(donations)
    else:
        sections = ''.join(wordString)

        # I'm doing the most recent donation amount outside the while loop bc regular expression is crazy good
        pattern = r'\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\d{4} [^\d,]+ (\d+)'

        # Split the sections by the "---------" delimiter
        sections = [item for item in [item.strip() for item in sections.split("---------")] if item]
        
        modified_sections = ""

        # section = "Most recent ...", "Presidential", "Issue", "Most recent House"
        for section in sections:
            # Dont include the presidential donations
            if "Presidential" in section:
                continue
            modified_sections = modified_sections + section

        # Getting individual donations using regex because regex is live laugh love                           
        pattern = r'\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\d{4} [^\d,]+ (\d+)'

        # Find all matches of the pattern in the text
        donations = re.findall(pattern, modified_sections)

        # Finds all the donation amounts in the text
        donation_amounts = [int(donation) for donation in donations]

        # Most recent donation
        most_recent_donation_amount = donation_amounts[-1] if donation_amounts else None

        # Sets most recent donation in df in column "Most Recent Give Amount"
        df_with_dummies.at[i, "Most Recent Give Amount"] = most_recent_donation_amount


        # Getting Max Dono Amount
        max_donation_amount = max(donation_amounts)

        # Setting Max Dono Amount at df in column "Max Give"
        df_with_dummies.at[i, "Max Give"] = max_donation_amount

        # Setting the average of the donation amounts in df in column "Average Give Amount"
        df_with_dummies.at[i, "Average Give Amount"] = sum(donation_amounts) / len(donation_amounts)


        # Pattern captures the month and the amount
        pattern_with_month = r'\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)(\d{4}) [^\d,]+ (\d+)'

        # Find all matches of the pattern in section
        matches_with_month = re.findall(pattern_with_month, modified_sections)

        # Converting matches to list of tuples with month, year, and amount
        donation_details = [(match[0], int(match[1]), int(match[2])) for match in matches_with_month]

        # Month to int conversion dictionary
        month_to_int = {
            'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
            'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
        }

        # Adjust the donation_details list to convert the month abbreviation to an integer
        donation_details_int_month = [(month_to_int[match[0]], match[1], match[2]) for match in donation_details]

        # Most recent donation details
        most_recent_donation_details = donation_details_int_month[-1]

        # current date
        current_date = datetime.now()

        # most recent donation date in datetime object
        donation_date = datetime(most_recent_donation_details[1], most_recent_donation_details[0], 1)

        # How long its been since last donation (in months)
        months_since_donation = (current_date.year - most_recent_dono.year) * 12 + (current_date.month - most_recent_dono.month)

        # set most_recent_dono into the dataframe
        df_with_dummies.at[i, "Months Since Last Donation"] = months_since_donation


        # Setting the length of donation detail as number of donations
        df_with_dummies.at[i, "Number of Gives"] = len(donation_details)
# print(df_with_dummies["Most Recent Give Amount"])
print(df[df_with_dummies["Average Give Amount"] == 0])
# print(df_with_dummies.at[190, "Bio Contributions"])