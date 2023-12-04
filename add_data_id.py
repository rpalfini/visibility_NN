import csv_file_combiner as cfc







if __name__ == "__main__":
    # fname = "./ML_code/data/main_data_file_courses1.csv"
    # output_fname = "./ML_code/data/23_03_15_main_data_file_courses1_id.csv"
    fname = "./ML_code/data/small_main_data_file_courses3.csv"
    output_fname = "./ML_code/data/small_main_data_file_courses3_id.csv"

    f = open(output_fname,"w")
    data_gen_in = cfc.csv_reader(fname)
    ii = 0
    for row in data_gen_in:
        tokens = row.split(",")
        tokens.append(str(ii))
        out_row = ",".join(tokens)
        f.write(out_row)
        ii += 1
    f.close()