"""
After extracting the RAR, we run this to move all the files into
the appropriate train/test folders.

Should only run this file once!
"""
import os
import os.path

def get_train_file(version='01'):
    train_file = os.path.join('data/TrainTestlist', 'trainlist' + version + '.txt')
    return train_file
def get_test_file(version='01'):
    test_file = os.path.join('data/TrainTestlist', 'testlist' + version + '.txt')
    return test_file

def generate_train_test_files(version='01'):
    """
    Dynamic generate train/test file as a name list txt file.
    """
    path_annotation = 'trainingset_annotations.txt'
    path_train = './data/rtvcdata/train'
    path_test = './data/rtvcdata/test'
    ## clear file.
    open(get_train_file(), 'w').close()
    open(get_test_file(), 'w').close()
    ###
    files_train = os.listdir(path_train)
    print("files_train: %s", files_train)
    for name in files_train:
        with open(path_annotation,'r') as fin:
            lines = fin.readlines()
            txt_file_train = open(get_train_file(), "a")
            for line in lines:
                if line.find(name)!=-1:
                    print("found match line: %s, name: %s, then write", line, name)
                ### write it to train list text file line by line
                    txt_file_train.write("%s" % line)
            txt_file_train.close()
    ###
    files_test = os.listdir(path_test)
    print("files_test: %s", files_test)
    for name in files_test:
        with open(path_annotation,'r') as fin:
            lines = fin.readlines()
            txt_file_test = open(get_test_file(), "a")
            for line in lines:
                if line.find(name)!=-1:
                    print("found match line: %s, name: %s, then write", line, name)
                ### write it to train list text file line by line
                    txt_file_test.write("%s" % line)
            txt_file_test.close()

def main():
    """
    Go through each of our train/test text files and move the videos
    to the right place.
    """
    ###!! Generate dataset at first.
    generate_train_test_files()

if __name__ == '__main__':
    main()
