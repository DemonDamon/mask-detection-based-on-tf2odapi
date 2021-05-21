import os  
import glob  
import pandas as pd 
import argparse
import xml.etree.ElementTree as ET  


ap = argparse.ArgumentParser()
ap.add_argument("-x", "--xml_files_path", type=str, required=True, help="path to .xml anntation files")
ap.add_argument("-o", "--csv_output_path", type=str, required=True, help="path to save output .csv file")
args = vars(ap.parse_args())


print("[INFO] Reading xml from folder {}".format(args["xml_files_path"]))
xml_list = []  
for xml_file in glob.glob(os.path.join(args["xml_files_path"], '*.xml')):  
    tree = ET.parse(xml_file)  
    root = tree.getroot()  
    for member in root.findall('object'):  
        value = (root.find('filename').text,  
                 int(root.find('size')[0].text),  
                 int(root.find('size')[1].text),  
                 member[0].text,  # object name
                 int(member.find("bndbox")[0].text),  
                 int(member.find("bndbox")[1].text),  
                 int(member.find("bndbox")[2].text),  
                 int(member.find("bndbox")[3].text)  
                 )  
        xml_list.append(value)
column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']  
xml_df = pd.DataFrame(xml_list, columns=column_name)
xml_df.to_csv(os.path.join(args["csv_output_path"], "annotations.csv"), index=None)
print("[INFO] Successfully converted xml to csv.")
