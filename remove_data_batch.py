import csv_file_combiner as cfc

fpaths = []
fpaths.append("C:/Users/Robert/git/visibility_NN/results_merge/23_02_19_merge.csv")
fpaths.append("C:/Users/Robert/git/visibility_NN/results_merge/23_02_18_19_20_merge.csv")
fpaths.append("C:/Users/Robert/git/visibility_NN/results_merge/23_02_18_and_19_merge.csv")

for path in fpaths:
    cfc.remove_invalid_data(path)


