"""
After extracting the RAR, we run this to move all the files into
the appropriate train/test folders.

Should only run this file once!
"""
import os
import os.path
#@see: https://www.techbeamers.com/python-copy-file/#shutil-copyfile
from shutil import copyfile
from sys import exit

def get_train_file(version='01'):
    train_file = os.path.join('data/TrainTestlist', 'trainlist' + version + '.txt')
    return train_file
def get_test_file(version='01'):
    test_file = os.path.join('data/TrainTestlist', 'testlist' + version + '.txt')
    return test_file

def get_train_test_lists(version='01'):
    """
    Using one of the train/test files (01, 02, or 03), get the filename
    breakdowns we'll later use to move everything.
    """
    # Get our files based on version.
    test_file = get_test_file(version)
    train_file = get_train_file(version)

    # Build the test list.
    with open(test_file) as fin:
        test_list = [row.strip() for row in list(fin)]
        test_list = [row.split(' ')[0] for row in test_list]

    # Build the train list. Extra step to remove the class index.
    with open(train_file) as fin:
        train_list = [row.strip() for row in list(fin)]
        train_list = [row.split(' ')[0] for row in train_list]

    # Set the groups in a dictionary.
    file_groups = {
        'train': train_list,
        'test': test_list
    }

    return file_groups

def move_files(file_groups):
    """This assumes all of our files are currently in _this_ directory.
    So move them to the appropriate spot. Only needs to happen once.
    """
    # Do each of our groups.
    for group, videos in file_groups.items():

        # Do each of our videos.
        for video in videos:
            print(video)
            # Get the parts.
            #parts = video.split(os.path.sep)
            parts = video.split(",")
            #print(parts)
            classname = parts[1]
            #classname = ""
            #print(classname)
            filename = parts[0]
            print(classname,filename)

            # Check if this class exists.
            # if not os.path.exists(os.path.join(group, classname)):
            #     print("Creating folder for %s/%s" % (group, classname))
            #     os.makedirs(os.path.join(group, classname))

            # Check if we have already moved this file, or at least that it
            # exists to move.
            fullfilename = "data/rtvcdata/" + group + "/"+ filename
            print(fullfilename)
            if not os.path.exists(fullfilename):
                print("Can't find %s to move. Skipping." % (fullfilename))
                continue

            # Move it.
            dest = os.path.join('data/',group, filename)
            print("Copying %s to %s" % (fullfilename, dest))
            #os.rename(fullfilename, dest)
            copyfile(fullfilename, dest)

    print("Done.")

def main():
    """
    Go through each of our train/test text files and move the videos
    to the right place.
    """
    # Get the videos in groups so we can move them.
    group_lists = get_train_test_lists()
    #print(group_lists)
    # Move the files.
    move_files(group_lists)

if __name__ == '__main__':
    main()
