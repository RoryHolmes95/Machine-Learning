#Subdirectory Sorting
#make sure this code is save IN the directory you want to sort out of bedlam will ensue
import os
from pathlib import Path
import shutil

subDirectories = {
    "DOCUMENTS": ['.pdf','.rtf','.txt'],
    "AUDIO": ['.m4a','.m4b','.mp3'],
    "VIDEOS": ['.mov','.avi','.mp4'],
    "IMAGES": ['.jpg','.jpeg','.png']
}


def pickDirectory(value):
    for category, suffixes in subDirectories.items():
        for suffix in suffixes:
            if suffix == value:
                return category
    return 'Misc' #if extension doesn't fit into ones above


def organizeDirectory(folder):
    for item in os.scandir(folder): #OS.scandir searches what is in current directory
        #print(item)
        if item.is_dir():
            continue #skips directorys which exist
        filePath = Path(item) #Path from lib above **
        print (filePath)
        filetype = filePath.suffix.lower() #isolates suffix, .suffix returns file extension
        directory = pickDirectory(filetype)
        pathName = os.path.join(folder, directory)
        directoryPath = Path(pathName)
        print (pathName)
        if directoryPath.is_dir() != True: #creates directory to move files into if it doesn't exist
            os.mkdir(pathName)  #makes the directory
        name = (os.path.basename(filePath))
        orig = filePath
        dest = pathName+'/' + name
        shutil.move(orig, dest)#moves file into directory by changing it's path to join with directory path

organizeDirectory()
