import json

def convert_json_to_text(json_file_path, output_file_path):
    # Read the JSON file
    with open(json_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    # Open the output file
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        # Iterate through each comment in the JSON data
        for comment in data:
            # Write the formatted text
            output_file.write(f"### {comment['video_title']}\n")
            output_file.write(f"{comment['x']}\n\n")

# Usage
json_file_path = 'comments.json'
output_file_path = 'formatted_comments.txt'

convert_json_to_text(json_file_path, output_file_path)
print(f"Conversion complete. Output written to {output_file_path}")