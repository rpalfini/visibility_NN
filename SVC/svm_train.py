from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

import config
from ML_code import util



def main(args):


    file_path = args.file_path
    split_percentages = [0.9, 0.05, 0.05]
    dataset = util.load_data(file_path)
    split_data = util.shuffle_and_split_data(dataset,args.num_obs,split_percentages)
        

    clf = SVC(kernel='rbf')
    clf.fit(split_data["X_train"],split_data["Y_train"])
    y_pred = clf.predict(split_data["X_test"])
    print(accuracy_score(split_data["Y_test"] ,y_pred))




if __name__ == "__main__":
    args = util.arg_parse()
    main(args)