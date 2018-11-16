"""
After moving all the files using the 1_ file, we run this one to extract
the images from the videos and also create a data file we can use
for training and testing later.
"""
import csv
import glob
import os
import os.path
from subprocess import call

def extract_files():
    """After we have all of our videos split between train and test, and
    all nested within folders representing their classes, we need to
    make a data file that we can reference when training our RNN(s).
    This will let us keep track of image sequences and other parts
    of the training process.

    We'll first need to extract images from each of the videos. We'll
    need to record the following data in the file:

    [train|test], class, filename, nb frames

    Extracting can be done with ffmpeg:
    `ffmpeg -i video.mpg image-%04d.jpg`
    """
    data_file = []
    folders = ['train', 'test']

    for folder in folders:
        class_folders = glob.glob(os.path.join('data', folder, '*'))
        #print(class_folders)

        for vid_class in class_folders:
            print("extract_files,vid_class:", vid_class)
            #class_files = glob.glob(os.path.join(vid_class, '*.mp4'))
            #class_files = vid_class

            #for video_path in class_files:
                # Get the parts of the file.
            video_parts = get_video_parts(vid_class)

            train_or_test, classname, filename_no_ext, filename = video_parts

            # Only extract if we haven't done it yet. Otherwise, just get
            # the info.
            if not check_already_extracted(video_parts):
                # Now extract it.
                src = os.path.join('data', train_or_test, filename)
                dest = os.path.join('data', train_or_test,
                    filename_no_ext + '-%04d.jpg')
                #print(src,dest)
                call(["ffmpeg", "-i", src, dest])

            # Now get how many frames it is.
            nb_frames = get_nb_frames_for_video(video_parts)

            data_file.append([train_or_test, classname, filename_no_ext, nb_frames])

            print("Generated %d frames for %s" % (nb_frames, filename_no_ext))

    ## clear file.
    open('data_file.csv', 'w').close()
    with open('data_file.csv', 'w') as fout:
        writer = csv.writer(fout)
        writer.writerows(data_file)

    print("Extracted and wrote %d video files." % (len(data_file)))

def get_nb_frames_for_video(video_parts):
    """Given video parts of an (assumed) already extracted video, return
    the number of frames that were extracted."""
    train_or_test, classname, filename_no_ext, _ = video_parts
    generated_files = glob.glob(os.path.join('data', train_or_test,
                                filename_no_ext + '*.jpg'))
    return len(generated_files)

def get_video_classes(video_name):
	"""find video classes from anotation txt file."""
	path_annotation = 'trainingset_annotations.txt'
	with open(path_annotation,'r') as fin:
		lines = fin.readlines()
		for line in lines:
			if line.find(video_name)!=-1:
				print("get_video_classes,found match line: %s, video_name: %s, then write", line, video_name)
				parts = line.split(",")
				classnames = "" # e.g: c1_c2_...
				for i in range(1, len(parts)):#without filename as index 0
					classnames += "_" + parts[i].strip("\n") 
				print("get_video_classes:",classnames)
				return classnames

def get_video_parts(video_path):
    """Given a full path to a video, return its parts."""
    parts = video_path.split(os.path.sep)
    #parts = video_path.split("\\")
    print("get_video_parts:",parts)
    #filename = parts[2].split("_")[0]#1000000031_4-0001.jpg
    filename = parts[2]
    print("get_video_parts,filename:",filename)
    classname = get_video_classes(filename)
    print("get_video_parts,classname:",classname)
    filename_no_ext = filename.split('.')[0] + classname
    print("get_video_parts,filename_no_ext:",filename_no_ext)
    
    train_or_test = parts[1]

    return train_or_test, classname, filename_no_ext, filename

def check_already_extracted(video_parts):
    """Check to see if we created the -0001 frame of this file."""
    train_or_test, classname, filename_no_ext, _ = video_parts
    return bool(os.path.exists(os.path.join(train_or_test, classname,
                               filename_no_ext + '-0001.jpg')))

def main():
    """
    Extract images from videos and build a new file that we
    can use as our data input file. It can have format:

    [train|test], class, filename, nb frames
    """
    extract_files()

if __name__ == '__main__':
    main()
