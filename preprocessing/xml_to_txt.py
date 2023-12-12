import os 
import tarfile
import shutil
import xml.dom.minidom

def convert(xml_file_path, frame_folder_path, label_folder_path):
    dom = xml.dom.minidom.parse(xml_file_path)
    root = dom.documentElement
    
    frame_list = root.getElementsByTagName('frame')
    label_name_txt_list = [file.replace('jpg','txt') for file in os.listdir(frame_folder_path)]
    
    for frame in frame_list:
        frame_number_str = frame.getAttribute('number')
        with open(os.path.join(label_folder_path, label_name_txt_list[int(frame_number_str)]), 'w') as txt_file:
                object_list = frame.getElementsByTagName('object')
                for obj in object_list:
                    obj_id = obj.getAttribute('id')
                    orientation = obj.getElementsByTagName('orientation')[0].firstChild.nodeValue
                    box_info = obj.getElementsByTagName('box')[0]
                    xc = box_info.getAttribute('xc')
                    yc = box_info.getAttribute('yc')
                    w = box_info.getAttribute('w')
                    h = box_info.getAttribute('h')
                    appearance = obj.getElementsByTagName('appearance')[0].firstChild.nodeValue

                    txt_file.write(f'{xc}, {yc}, {w}, {h}, ')

                    # Extract and write information for each hypothesis in the object
                    hypothesis_list = obj.getElementsByTagName('hypothesis')
                    for hypothesis in hypothesis_list:
                        hypothesis_id = hypothesis.getAttribute('id')
                        role = hypothesis.getElementsByTagName('role')[0].firstChild.nodeValue
                        #txt_file.write(f'{obj_id}, {role}')
                        txt_file.write(f'{role}')
                    txt_file.write('\n')  # Add a newline between objects

        print("Text files generated successfully.")

if __name__ == "__main__":
    os.chdir('D:\code_folder\data-code\Datathon2023\code\data') #data_root 
    data_dir = os.listdir()
    cor_dir = os.path.join(data_dir[0], 'cor')
    front_dir = os.path.join(data_dir[0], 'front')
    cor_list = os.listdir(cor_dir)
    front_list = os.listdir(front_dir)
    for dir_ in cor_list:
        root = os.path.join(cor_dir, dir_)
        file = dir_ + '.xml'
        xml_file_path = os.path.join(root, file)
        frame_folder_path = os.path.join(root, 'frame')
        label_folder_path = os.path.join(root, 'label')
        os.makedirs(label_folder_path, exist_ok=True)
        convert(xml_file_path, frame_folder_path, label_folder_path)
    for dir_ in front_list:
        root = os.path.join(front_dir, dir_)
        file = dir_ + '.xml'
        xml_file_path = os.path.join(root, file)
        frame_folder_path = os.path.join(root, 'frame')
        label_folder_path = os.path.join(root, 'label')
        os.makedirs(label_folder_path, exist_ok=True)
        convert(xml_file_path, frame_folder_path, label_folder_path)