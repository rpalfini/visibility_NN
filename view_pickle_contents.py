# can be used to create default args for problems with arguments
import pickle

def view_pickle_contents(pickle_file_path):

    try:
        with open(pickle_file_path, 'rb') as file:
            loaded_variable = pickle.load(file)
            # Now, loaded_variable contains the data you saved in the Pickle file
            print("Variable loaded successfully:", loaded_variable)
    except FileNotFoundError:
        print(f"File {pickle_file_path} not found.")
    except Exception as e:
        print("An error occurred while loading the Pickle variable:", e)


if __name__ == "__main__":
    pickle_file_path = "boundary_args.pickle"
    view_pickle_contents(pickle_file_path)