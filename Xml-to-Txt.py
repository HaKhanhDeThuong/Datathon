import xml.etree.ElementTree as ET

# Load the XML file
tree = ET.parse('OneStopNoEnter1.xml')
root = tree.getroot()

# Iterate through each frame in the specified range
for frame_number in range(725):  
    frame_number_str = str(frame_number)
    
    # Check if a frame with the current number exists in the XML
    frame = root.find(f'.//frame[@number="{frame_number_str}"]')
    if frame is not None:
        # Open a text file for writing
        with open(f'OneStopNoEnter1cor{frame_number}.txt', 'w') as txt_file:
            #txt_file.write(f'Frame Number: {frame_number_str}\n')

            # Extract and write information for each object in the frame
            for obj in frame.findall('.//object'):
                obj_id = obj.get('id')
                orientation = obj.find('orientation').text
                box_info = obj.find('box')
                xc = box_info.get('xc')
                yc = box_info.get('yc')
                w = box_info.get('w')
                h = box_info.get('h')
                appearance = obj.find('appearance').text

                # txt_file.write(f'{obj_id}\n')
                # txt_file.write(f'{orientation}\n')
                txt_file.write(f'{xc}, {yc}, {w}, {h}\n')
                # txt_file.write(f'{appearance}\n')

                # Extract and write information for each hypothesis in the object
                for hypothesis in obj.findall('.//hypothesis'):
                    hypothesis_id = hypothesis.get('id')
                    prev = hypothesis.get('prev')
                    evaluation = hypothesis.get('evaluation')
                    movement = hypothesis.find('movement').text
                    role = hypothesis.find('role').text
                    context = hypothesis.find('context').text
                    situation = hypothesis.find('situation').text

                    # txt_file.write(f'{hypothesis_id}\n')
                    # txt_file.write(f'{prev}\n')
                    # txt_file.write(f'{evaluation}\n')
                    txt_file.write(f'{movement}\n')
                    txt_file.write(f'{role}\n')
                    txt_file.write(f'{context}\n')
                    txt_file.write(f'{situation}\n')

                txt_file.write('\n')  # Add a newline between objects

print("Text files generated successfully.")