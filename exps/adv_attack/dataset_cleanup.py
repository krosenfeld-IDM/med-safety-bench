# Script to remove duplicates from the dataset
import argparse
import random
import os

parser = argparse.ArgumentParser(description='Remove duplicates from the dataset and save the cleaned dataset to the dedup directory.')
parser.add_argument('data_dir', type=str, help='Path to the data directory containing the files to be cleaned.')
args = parser.parse_args()

data_dir = args.data_dir

# Set to store unique lines across all files
unique_lines = set()

# Read all files in the data directory
files = [f for f in os.listdir(data_dir) if f.startswith('category_') and f.endswith('.txt')]
files.sort()

print('Files:\n')
for file in files:
    print('\t' + file)

all_lines = []
for file in files:
    file_path = os.path.join(data_dir, file)

    with open(file_path, 'r') as f:
        for item in f.readlines():
            all_lines.append({'file': file, 'line': item})

# Shuffle the lines
random.shuffle(all_lines)

# Remove duplicates
file_lines = {}
num_duplicates = {}
for item in all_lines:
    if item['line'] not in unique_lines:
        unique_lines.add(item['line'])
        if item['file'] not in file_lines:
            file_lines[item['file']] = []
        file_lines[item['file']].append(item['line'])
    else:
        if item['file'] not in num_duplicates:
            num_duplicates[item['file']] = 0
        num_duplicates[item['file']] += 1

# Sort keys
file_lines = dict(sorted(file_lines.items()))
num_duplicates = dict(sorted(num_duplicates.items()))

print('\nNumber of duplicates:')
for file, num in num_duplicates.items():
    print(f'\t{file}: {num}')
print('\n\tTotal:', sum(num_duplicates.values()))

print('\nNumber of unique lines:')
for file, lines in file_lines.items():
    print(f'\t{file}: {len(lines)}')
print('\n\tTotal:', sum([len(lines) for lines in file_lines.values()]))

output_dir = os.path.join(data_dir, 'dedup')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Write the cleaned dataset to data file
for file, lines in file_lines.items():
    with open(os.path.join(output_dir, file), 'w') as f:
        for item in lines:
            f.write(item)
