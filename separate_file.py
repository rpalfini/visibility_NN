
large_file = "./ML_code/Data/main_data_file_courses3.csv"
small_file = "./ML_code/Data/small_main_data_file_courses3.csv"
with open(large_file, 'r') as source_file:
    # Open a new CSV file in write mode
    with open(small_file, 'w') as new_file:
        # Loop through the first 25,000 lines
        for line_number, line in enumerate(source_file):
            if line_number < 25000:
                # Write the line to the new file
                new_file.write(line)
            else:
                break  # Stop when 25,000 lines are copied

print("Copy operation complete.")
