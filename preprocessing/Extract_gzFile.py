import os 
import tarfile
import shutil

def extractData(dir_, list_dir):
    for item in list_dir:
        if '.gz' in item:
            file = os.path.join(dir_, item)
            folder = os.path.join(dir_, item.split('.')[0])
            folder_frame = os.path.join(folder, 'frame')
            os.makedirs(folder, exist_ok=True)
            os.makedirs(folder_frame, exist_ok=True)
            with tarfile.open(file, 'r:gz') as tar:
                tar.extractall(folder_frame)
            shutil.move(os.path.join(dir_,item.split('.')[0]+'.xml'), folder)
            shutil.move(os.path.join(dir_,item.split('.')[0]+'.mpg'), folder)
            os.remove(file)
            print(f'extrace {file} success !!!')

if __name__ == "__main__":
    os.chdir('D:\code_folder\data-code\Datathon2023\git\Datathon\data')
    data_dir = os.listdir()
    cor_dir = os.path.join(data_dir[0], 'cor')
    front_dir = os.path.join(data_dir[0], 'front')
    extractData(cor_dir, os.listdir(cor_dir))
    extractData(front_dir, os.listdir(front_dir))