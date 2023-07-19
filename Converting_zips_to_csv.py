import os
import pandas as pd
import gzip

# # Making a loop to open all the ZIP files
# # Set your directory
# dir_name = '/Users/rishabhgoel/Desktop/Projects for Resume/Single Cell Lung Atlas Project'
#
# # Get list of all files in directory
# file_list = os.listdir(dir_name)
#
# # Filter list down to .gz files
# gz_files = [file for file in file_list if file.endswith('.gz')]
#
# chunksize = 10 ** 6  # adjust this value to suit your available memory
#
# # For each .gz file, decompress it and read it into a pandas DataFrame
# for gz_file in gz_files:
#     csv_file = gz_file[:-3]  # Remove .gz from the end
#     csv_file_path = os.path.join(dir_name, csv_file)
#
#     # Create or overwrite the csv file
#     with open(csv_file_path, 'w') as f_out:
#         pass
#
#     # Process chunks
#     with gzip.open(os.path.join(dir_name, gz_file), 'rt') as f_in:
#         for chunk in pd.read_csv(f_in, chunksize=chunksize):
#             # You can now work with the dataframe `chunk`...
#             # For instance, append chunk to csv file:
#             with open(csv_file_path, 'a') as f_out:
#                 chunk.to_csv(f_out, index=False)
#
#




# with gzip.open('/Users/rishabhgoel/Desktop/Gen1E-RIDGE/Mimic v1.0/diagnoses_icd.csv.gz', 'rt') as f:
#     df = pd.read_csv(f)
#
# df.to_csv('/Users/rishabhgoel/Desktop/Gen1E-RIDGE/Mimic v1.0/diagnoses_icd.csv', index=False)




# import csv
# from tqdm import tqdm
#
# input_file = '/Users/rishabhgoel/Desktop/Gen1E-RIDGE/MIMIC-IV Project and eICU-CRD/mimic-iv-2.2/icu/chartevents.csv'
# output_file = '/Users/rishabhgoel/Desktop/Gen1E-RIDGE/MIMIC-IV Project and eICU-CRD/mimic-iv-2.2/icu/chartevents_revised.csv'
#
# # Get the total number of lines in the input file
# total_lines = sum(1 for _ in open(input_file))
#
# with open(input_file, 'r') as csv_file, open(output_file, 'w') as output:
#     reader = csv.reader(csv_file)
#     writer = csv.writer(output)
#     header = next(reader)
#     writer.writerow(header)
#
#     # Use tqdm to create a progress bar
#     progress_bar = tqdm(reader, total=total_lines-1)  # Subtract 1 for the header line
#
#     for row in progress_bar:
#         # Process each row and write to the output file
#         writer.writerow(row)
#
# # Close the progress bar once the processing is complete
# progress_bar.close()


import pandas as pd
import gzip

chunksize = 10 ** 6
file_path = '/Users/rishabhgoel/Desktop/Gen1E-RIDGE/MIMIC-IV Project and eICU-CRD/eicu-collaborative-research-database-2.0/nurseCharting.csv.gz'
output_path = '/Users/rishabhgoel/Desktop/Gen1E-RIDGE/MIMIC-IV Project and eICU-CRD/eicu-collaborative-research-database-2.0/nurseCharting.csv'

# Create or overwrite the csv file
with open(output_path, 'w') as f_out:
    pass

# Process chunks
with gzip.open(file_path, 'rt') as f_in:
    for chunk in pd.read_csv(f_in, chunksize=chunksize):
        # Append chunk to csv file:
        with open(output_path, 'a') as f_out:
            chunk.to_csv(f_out, index=False)
