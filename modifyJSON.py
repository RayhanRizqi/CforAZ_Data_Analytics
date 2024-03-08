import json

# Read the existing JSON file
json_file_path = "committee_candidate.json"

with open(json_file_path, 'r') as json_file:
    json_records = json.load(json_file)

# Create a dictionary to store the new format
new_json_data = {}

# Iterate through the existing records and organize them by committee_name
for record in json_records:
    committee_id = record['committee_id']

    # Create or update the entry in the new dictionary
    if committee_id not in new_json_data:
        new_json_data[committee_id] = []

    # Append the relevant values to the list
    new_json_data[committee_id].append({
        'committee_name': record['committee_name'],
        'candidate_name': record['candidate_name'],
        'candidate_id': record['candidate_id'],
        'party_affiliation': record['party_affiliation']
    })

# Write the new JSON records to a file
new_json_file_path = "committee_candidate_Rayhan_Nivedh.json"

with open(new_json_file_path, 'w') as new_json_file:
    json.dump(new_json_data, new_json_file, indent=2)

print(f"New JSON file created at: {new_json_file_path}")
