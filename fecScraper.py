import requests
import json

# Set the base URL for the OpenFEC API
base_url = "https://api.open.fec.gov/v1/"

# Set your API key (you can obtain it by registering on the OpenFEC website)
api_key = "x7cBwQEPuoclyOMqvvrbSvsAdEjJNfXMCWZMQsRu"

# Define a function to make API requests
def make_api_request(endpoint, params=None):
    url = base_url + endpoint
    params = params or {}
    params["api_key"] = api_key

    response = requests.get(url, params=params)

    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error {response.status_code}: {response.text}")
        return None

# Load the modified JSON file
json_file_path = "committee_candidate_Rayhan_Nivedh.json"
with open(json_file_path, 'r') as json_file:
    committee_candidate_data = json.load(json_file)

# Example: Get a large dataset of individual contributions
contributions_endpoint = "schedules/schedule_a/"
contributions_params = {
    "two_year_transaction_period": 2024,  # Adjust the two-year period as needed
    "per_page": 100,  # Adjust the number of results per page as needed
    "page": 1  # Start with the first page
}

target_donors = 50  # Set the target number of donors
processed_donors = 0  # Initialize the processed donors counter

# Iterate through all pages
for page in range(1, 100):  # Assuming you have fewer than 100 pages
    contributions_params['page'] = page

    print(f"\nProcessing page {page}...")

    contributions_data = make_api_request(contributions_endpoint, params=contributions_params)

    if contributions_data and 'results' in contributions_data:
        contributions = contributions_data['results']
        dem_count = 0
        rep_count = 0
        other_count = 0

        if contributions:
            # Aggregate contributions by donor
            for contribution in contributions:
                donor_name = contribution.get('contributor_name', 'Unknown')
                committee_id = contribution.get('committee_id')

                try:
                    # Check if donor information is available in the local JSON file
                    committee = committee_candidate_data[committee_id]
                    # democratic_amount = affiliations['DEM']
                    # republican_amount = affiliations['REP']
                    # unknown_amount = affiliations['Unknown']
                    if committee[0]['party_affiliation'] == "DEM":
                        dem_count += 1
                    elif committee[0]['party_affiliation'] == "REP":
                        rep_count += 1
                    else:
                        other_count += 1
                except KeyError:
                    print(f"Committee ID {committee_id} not found in the local JSON file.")

                # print(f"\nDonor: {donor_name}")
                # print(f"Democratic Contributions: ${democratic_amount}")
                # print(f"Republican Contributions: ${republican_amount}")
                # print(f"Unknown Contributions: ${unknown_amount}")

                processed_donors += 1

                if processed_donors >= target_donors:
                    print(f"\nTarget number of donors ({target_donors}) reached. Stopping.")
                    break

        else:
            print(f"No contributions found on page {page}.")

        if processed_donors >= target_donors:
            break
    else:
        print(f"Error fetching contributions on page {page}.")

    if processed_donors >= target_donors:
        break
