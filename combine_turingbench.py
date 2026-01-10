#!/usr/bin/env python3
"""Combine TuringBench files into a single file."""

import os

# Define input and output paths
data_dir = 'Data'
input_files = [
    os.path.join(data_dir, 'TuringBench_1.txt'),
    os.path.join(data_dir, 'TuringBench_2.txt'),
    os.path.join(data_dir, 'TuringBench_3.txt')
]
output_file = os.path.join(data_dir, 'TuringBench.txt')

print(f"Combining TuringBench files...")

# Open output file and write all input files to it
with open(output_file, 'w', encoding='utf-8') as outfile:
    for idx, input_file in enumerate(input_files, 1):
        if os.path.exists(input_file):
            print(f"  Reading {input_file}...")
            with open(input_file, 'r', encoding='utf-8') as infile:
                content = infile.read()
                outfile.write(content)
                # Add newline between files if the last line doesn't have one
                if content and not content.endswith('\n'):
                    outfile.write('\n')
        else:
            print(f"  Warning: {input_file} not found, skipping...")

print(f"\nCombined file created: {output_file}")
print("Done!")
