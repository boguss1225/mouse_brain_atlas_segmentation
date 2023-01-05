import cv2   
import os
import glob

srcpath = 'database/IMAS_Salmon/train_annotation/' # Source Folder
dstpath = 'database/IMAS_Salmon/train_annotation_qt/' # Destination Folder

def divide_into_4pics(file_dir): 
    # load image
    img = cv2.imread(file_dir)
    base=os.path.basename(file_dir)
    filename = os.path.splitext(base)[0]
    extention = os.path.splitext(base)[1]

    ##########################################
    # At first vertical devide image         #
    ##########################################
    # start vertical devide image
    height = img.shape[0]
    width = img.shape[1]
    # Cut the image in half
    width_cutoff = width // 2
    left1 = img[:, :width_cutoff]
    right1 = img[:, width_cutoff:]
    # finish vertical devide image

    ##########################################
    # At first Horizontal devide left1 image #
    ##########################################
    #rotate image LEFT1 to 90 CLOCKWISE
    img = cv2.rotate(left1, cv2.ROTATE_90_CLOCKWISE)
    # start vertical devide image
    height = img.shape[0]
    width = img.shape[1]
    # Cut the image in half
    width_cutoff = width // 2
    l2 = img[:, :width_cutoff]
    l1 = img[:, width_cutoff:]
    # finish vertical devide image
    #rotate image to 90 COUNTERCLOCKWISE
    l2 = cv2.rotate(l2, cv2.ROTATE_90_COUNTERCLOCKWISE)
    #save
    #print("filename:"+filename)
    #print("extention:"+extention)
    filename1 = filename+"_1"+extention
    cv2.imwrite(os.path.join(dstpath,filename1), l2)
    #rotate image to 90 COUNTERCLOCKWISE
    l1 = cv2.rotate(l1, cv2.ROTATE_90_COUNTERCLOCKWISE)
    #save
    filename2 = filename+"_2"+extention
    cv2.imwrite(os.path.join(dstpath,filename2), l1)

    ##########################################
    # At first Horizontal devide right1 image#
    ##########################################
    #rotate image RIGHT1 to 90 CLOCKWISE
    img = cv2.rotate(right1, cv2.ROTATE_90_CLOCKWISE)
    # start vertical devide image
    height = img.shape[0]
    width = img.shape[1]
    # Cut the image in half
    width_cutoff = width // 2
    r4 = img[:, :width_cutoff]
    r3 = img[:, width_cutoff:]
    # finish vertical devide image
    #rotate image to 90 COUNTERCLOCKWISE
    r4 = cv2.rotate(r4, cv2.ROTATE_90_COUNTERCLOCKWISE)
    #save
    filename3 = filename+"_3"+extention
    cv2.imwrite(os.path.join(dstpath,filename3), r4)
    #rotate image to 90 COUNTERCLOCKWISE
    r3 = cv2.rotate(r3, cv2.ROTATE_90_COUNTERCLOCKWISE)
    #save
    filename4 = filename+"_4"+extention
    cv2.imwrite(os.path.join(dstpath,filename4), r3)

#@iterate folder
def iteration_trigger(folder_dir):
    explore_path = folder_dir + "/*.png"
    path_list = glob.glob(explore_path)
    print(explore_path)
    print("len",len(path_list))
    for file_dir in path_list:
        print(file_dir)
        divide_into_4pics(file_dir)

copy_bool = input("Copy target directory before excute. Excute? (Y/N):")

if copy_bool=='y' or copy_bool=='Y':
    print("Start conversion")
    iteration_trigger(srcpath)
    print("Done")
else : 
    print("See you")