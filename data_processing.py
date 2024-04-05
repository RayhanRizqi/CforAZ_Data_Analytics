import pandas as pd
import re
from datetime import datetime
import matplotlib.pyplot as plt

df = pd.read_csv("CFAZ Modeling Data.csv")

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

# Columns for reasons for Conor to give in list
reason_to_give = [
    "strong envir support", "very pro-gun control", "dems in red seats",
    "house races", "gives in primaries", "supports AZ campaigns", "strong pp support",
    "moderate dems"
    ]

# Convert reasons_to_give into columns in the df
for item in reason_to_give:
    df[item] = 0

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
# ethnicity_dummies = pd.get_dummies(df['Ethnicity'], prefix='Ethnicity')

# Concatenate these new one-hot encoded columns with the original DataFrame
df_with_dummies = pd.concat([df, gender_dummies], axis=1)
# df_with_dummies = pd.concat([df_with_dummies, ethnicity_dummies], axis=1)

# Prettiy Gender columns
for col in df_with_dummies.columns:
    if "Gender_" in col:
        newCol = col.split("_")
        df_with_dummies = df_with_dummies.rename(columns={col: newCol[1]})

# very questionable variable name, might as well call it ethnic Cleansing table
ethnicConversions = {"afam": "isAfam", "chinese": "isAsian", "korean": "isAsian",
                     "indian": "isAsian", "iranian": "isAsian", "japanese": "isAsian",
                     "muslim": "isAsian", "white": "isWhite", "greek": "isWhite",
                     "italian": "isWhite", "irish": "isIrish", "jewish": "isJewish",
                     "latinx": "isLatinx"}

# Strips the "likely" and "american" in Ethnicity column
df_with_dummies['Ethnicity'] = df_with_dummies['Ethnicity'].apply(lambda row: row.strip().split(" ")[1])

# It creates a column for every grouped ethnicity 
for value in set(ethnicConversions.values()):
    # Create a new column for each unique value
    if value not in df_with_dummies.columns:
        df_with_dummies[value] = 0

# add another column for "ethnicity unsure"
df_with_dummies["Ethnicity unsure"] = 0

# One hot encoding grouped ethnicities
for i in range(len(df_with_dummies)):
    ethnicity = df_with_dummies.at[i, "Ethnicity"]

    # if gender unsure, leave 
    if ethnicity == "unsure":
        df_with_dummies.at[i, "Ethnicity unsure"] = 1
        continue
    if "/" in ethnicity:
        ethnicities = ethnicity.split('/')
        for eth in ethnicities:
            df_with_dummies.at[i, ethnicConversions[eth]] = 1
        continue
        
    df_with_dummies.at[i, ethnicConversions[ethnicity]] = 1

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

# regex pattern for reasons to give in Giving Summary column
reason_to_give_pattern = r'reason_to_give: (.*)'

# Goes through each row of "Giving summary"
for i in range(len(df_with_dummies)):
    reasons_text = df_with_dummies.iloc[i]["Giving Summary"]

    #applies the regex pattern to look for matches
    reasons_match = re.search(reason_to_give_pattern, reasons_text)
    if reasons_match:
        # separates each reason into an item in a list
        reasons_list = [item for item in re.split(r',\s*', reasons_match.group(1))]

    # if the reason is one of the columns in df_with_dummies, change it to a 1
    for reason in reasons_list:
        if reason in df_with_dummies.columns:
            df_with_dummies.at[i, reason] = 1


# df_with_dummies["New Networth"] = df_with_dummies["Networth"] / 10000
# df_with_dummies["New Networth"].hist(bins = 30)
# # plt.xscale('log')

# plt.show()







# print(df_with_dummies[df_with_dummies["Networth"] == 96407])
# print(len(df_with_dummies[df_with_dummies["isLatinx"] == 1]))

columns_to_keep = ['isAfam', 'isAsian', 'isWhite', 'isIrish', 'isJewish',
       'isLatinx', 'Ethnicity unsure']
column_sums = df_with_dummies[columns_to_keep].sum()

column_sums.plot(kind='bar')
plt.xticks(rotation=45)  # Optional: Rotate the x-axis labels for better readability
plt.show()