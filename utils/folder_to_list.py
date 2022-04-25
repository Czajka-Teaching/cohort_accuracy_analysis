import os 
import argparse
from tqdm import tqdm
import numpy as np

#Given a metdatalist - such as txt with with list of folder all of which belongs to the same class
def one_metadata_to_list(metadatalist,sourcepath,label):
    print("started!")
    all_list = []
    load_metadata = np.loadtxt(metadatalist,dtype=str)
    for items in tqdm(load_metadata):
        folder_path = os.path.join(sourcepath,items)
        all_imgs_folder = os.listdir(folder_path)
        for images in all_imgs_folder:
            if images.endswith(".jpg"):
                all_list.append(os.path.join(folder_path,images)+" "+label)
    return all_list

#Given a metdatalist - such as txt with with list of folder - with each folder belonging to different class
def mult_metadata_to_list(sourcepath,label):
    pass

#Given a source path, could be with multiple directories - but all belonging to the same class
def folder_to_one_label(sourcepath,label):
    print("started!")
    all_list = []
    for subdir, dirs, files in tqdm(os.walk(sourcepath)):
        for file in files:
            filepath = os.path.join(os.path.abspath(subdir),file)
            #print(filepath)
            if filepath.endswith(".jpg") or filepath.endswith(".JPG") :
                all_list.append(filepath +" "+label)
    return all_list

#Given a source path with multiple directories, with each directory belonging to the object of different class
def folder_to_multiple_label(sourcepath,label):
    pass

           
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="List of metadatafolder/sourcepath to one label/source path to multilabel")
    parser.add_argument("--source", "-s", help="source folder")
    parser.add_argument("--destination", "-d", help="destination folder")
    # Gender classification is a binary class problem so assign label of 0 and 1 for each class. For multi-class some version of the code can be used
    parser.add_argument("--label", "-l", help="Label") #label=1 for female and 0 for male
    parser.add_argument("--name", "-n", help="Name of the file to save")
    parser.add_argument("--path", nargs='?', default='/afs/crc.nd.edu/user/a/abhatta/MLproject/ms1m_v2/extracted/', help="Main Crc Path")
    parser.add_argument("--option", "-opt",type=int, help="[0,1,2,3]")
    args = parser.parse_args()

    assert args.option in [0,1,2,3]


    ####### Options Explained  ##########
    '''
    Option = 0 -> if you have a list of folder all of with belong to same class
    Option = 1 -> For a source folder (could be multiple subdirectories) - with all belonging to same class
    #Future Additions
    Option = 2 -> if you have a list of folder that belongs to different class
    Option = 3 -> For a source folder (could be multiple subdirectories) - with each subdirectory belonging to one class
    '''
    ######################################

    if args.option == 0:
        imgpath_list = one_metadata_to_list(metadatalist=args.source,sourcepath=args.path,label = args.label)
    elif args.option ==1:
        imgpath_list = folder_to_one_label(sourcepath=args.source,label=args.label)
    elif args.option ==2:
        imgpath_list = folder_to_multiple_label(args.sourcepath,args.label)

    if not os.path.exists(args.destination):
        os.makedirs(args.destination)

    save_path = os.path.join(args.destination,args.name)

    np.savetxt("{}.txt".format(save_path),imgpath_list,fmt='%s',newline='\n')
