import os
import argparse
import pandas as pd


ap = argparse.ArgumentParser()
ap.add_argument("-p", "--csv_path", type=str, required=True, help="Path to annotation CSV file")
ap.add_argument("-o", "--save_path", type=str, default="./", help="")
args = vars(ap.parse_args())

data = pd.read_csv(args["csv_path"])

labelmap_string = ""
for _id, class_name in enumerate(set(data["class"])):
    labelmap_string += f"""
item {"{"}
    id: {_id+1}
    name: '{class_name}'
{"}"}\n"""

with open(os.path.join(args["save_path"], "labelmap.pbtxt"), 'w') as file:
    file.write(labelmap_string.strip())

print("[INFO] successfully saved labelmap file in => {}".format(os.path.join(args["save_path"], "labelmap.pbtxt")))
