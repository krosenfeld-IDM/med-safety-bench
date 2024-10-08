# Script to extract harmful prompts from the raw model outputs
import re
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('input_file', type=str, help='Path to the input file containing the raw model outputs')
parser.add_argument('output_file', type=str, help='Path to the output file to write the extracted prompts')
args = parser.parse_args()

input_file = args.input_file
output_file = args.output_file

# Read the input file
with open(input_file, 'r') as file:
    lines = file.readlines()

strip_chars = '\"*:\n '

# Open the output file for writing
with open(output_file, 'w') as outfile:
    for line in lines:
        # Check if the line matches the pattern of digits followed by a period and space
        if re.match(r'^\d+\.\s', line):
            # Extract the idea text (excluding the number and period)
            prompt = line.split(' ', 1)[1].strip(strip_chars)
            if '**' in prompt:
                prompt = prompt.split('**')[1].strip(strip_chars)
            if prompt == '':
                continue
            # Write the idea to the output file
            outfile.write(prompt + '\n')
