import os
import argparse
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


ap = argparse.ArgumentParser()
ap.add_argument("-p", "--csv_path", type=str, required=True, help="Path to annotation CSV file")
ap.add_argument("-s", "--test_size", type=float, default=0.3, help="Test size of split dataset")
ap.add_argument("-o", "--save_path", type=str, default="./", help="")
args = vars(ap.parse_args())

data = pd.read_csv(args["csv_path"])
train_inds, test_inds = next(GroupShuffleSplit(test_size=args["test_size"], random_state=7).split(data, groups=data['filename']))

train = data.iloc[train_inds]
test = data.iloc[test_inds]

train.to_csv(os.path.join(args["save_path"], "train_labels.csv"))
test.to_csv(os.path.join(args["save_path"], "test_labels.csv"))


