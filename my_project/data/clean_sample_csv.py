import csv

input_file = "data/sample.csv"
output_file = "data/sample_cleaned.csv"

with open(input_file, newline='', encoding='utf-8') as infile, \
     open(output_file, 'w', newline='', encoding='utf-8') as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile)
    for i, row in enumerate(reader):
        if i == 0:
            # Write the header as is
            writer.writerow(row)
        else:
            # Write only the last 8 columns (matching the header)
            writer.writerow(row[-8:])

# Optionally, replace the original file
import shutil
shutil.move(output_file, input_file)