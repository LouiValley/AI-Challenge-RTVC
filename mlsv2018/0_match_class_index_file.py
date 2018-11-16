"""
After extracting the RAR, we run this to move all the files into
the appropriate train/test folders.

Should only run this file once!
"""
import os
import os.path

def get_ucf101_file():
    ucf_file = os.path.join('data/TrainTestlist', 'classInd_ucf101' + '.txt')
    return ucf_file
def get_mlsv2018_file():
    mlsv_file = os.path.join('data/TrainTestlist', 'classInd' + '.txt')
    return mlsv_file

def get_mlsv2018_file_match():
    mlsv_file = os.path.join('data/TrainTestlist', 'classInd_match'  + '.txt')
    return mlsv_file

def match_class_index_file():
    """
    Dynamic generate class index file compare with UCF101's file.
    """
    ## clear file.
    txt_file_mlsv_match = open(get_mlsv2018_file_match(), 'w').close()
    ###
    with open(get_mlsv2018_file(),'r') as fin_mlsv:
        lines_mlsv = fin_mlsv.readlines()
        print("lines_mlsv:",lines_mlsv)
        for line_mlsv in lines_mlsv:
            with open(get_ucf101_file(),'r') as fin_ucf:
                lines_ucf = fin_ucf.readlines()
                print("lines_ucf:",lines_ucf)    
                for line_ucf in lines_ucf:
                    line_mlsv_name = line_mlsv.split(" ")[1]
                    line_ucf_index = line_ucf.split(" ")[0]
                    txt_file_mlsv_match  = open(get_mlsv2018_file_match(),'a')
                    if line_ucf.find(line_mlsv_name)!=-1:
                        print("found match line_ucf:",line_ucf
                            ,',line_mlsv_name:',line_mlsv_name
                            ,',line_ucf_index:', line_ucf_index )
                        line_mlsv_matched = line_mlsv +" "+ line_ucf_index
                        print("line_mlsv_matched:",line_mlsv_matched)
                    ### write it to train list text file line by line
                        txt_file_mlsv_match.write("%s" % line_mlsv_matched)
    txt_file_mlsv_match.close()

def main():
    """
    Go through each of our train/test text files and move the videos
    to the right place.
    """
    ###!! Generate dataset at first.
    match_class_index_file()

if __name__ == '__main__':
    main()
