import json
import csv
from io import StringIO

# Load JSON data from file
with open('C:/tareas/LLM/hackathon/training.json', 'r') as json_file:
    data = json.load(json_file)

csv_buffer = StringIO()

csv_writer = csv.writer(csv_buffer)

csv_writer.writerow(['instruction', 'input', 'output'])

for item in data:
    instruction = item['instruction']
    
    input_data = item['input']
    input_str = json.dumps(input_data)  # Convert input dictionary to JSON string
    
    output_data = item['output']
    output_str = json.dumps(output_data)  # Convert output dictionary to JSON string
    
    csv_writer.writerow([instruction, input_str, output_str])

csv_content = csv_buffer.getvalue()

with open('packets.csv', 'w', newline='') as csv_file:
    csv_file.write(csv_content)

print(csv_content)
