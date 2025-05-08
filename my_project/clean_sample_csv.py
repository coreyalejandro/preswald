import csv
import shutil

input_file = "data/sample.csv"
output_file = "data/sample_cleaned.csv"

def is_valid_row(row):
    # Check for exactly 8 columns and Row ID is a number
    return (
        len(row) == 8 and
        row[0].isdigit()
    )

with open(input_file, newline='', encoding='utf-8') as infile, \
     open(output_file, 'w', newline='', encoding='utf-8') as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile)
    for i, row in enumerate(reader):
        if i == 0:
            # Write header without 'Row ID'
            writer.writerow(row[1:])
        else:
            cleaned_row = row[-8:]
            if is_valid_row(cleaned_row):
                # Clean Product Name (index 1 after removing Row ID)
                data_row = cleaned_row[1:]
                if len(data_row) > 1:
                    # Remove single quotes, dashes, and single quote before N
                    product_name = data_row[1]
                    product_name = product_name.replace("-", "")
                    product_name = product_name.replace("'N", "N")
                    product_name = product_name.replace("'", "")
                    data_row[1] = product_name
                writer.writerow(data_row)

# Replace the original file with the cleaned file
shutil.move(output_file, input_file) 