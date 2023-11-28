'''creates a smaller version of a test file by grabbing the first num_lines of the file'''
large_file = "D:/Vis_network_data/data_file_by_course/main_data_file_courses20.csv"
small_file = "./ML_code/Data/small_main_data_file_courses20.csv"
# num_lines = 25000
num_lines = 250
with open(large_file, 'r') as source_file:
    # Open a new CSV file in write mode
    with open(small_file, 'w') as new_file:
        # Loop through the first 25,000 lines
        for line_number, line in enumerate(source_file):
            if line_number < num_lines:
                # Write the line to the new file
                new_file.write(line)
            else:
                break  # Stop when 25,000 lines are copied

print("Copy operation complete.")
