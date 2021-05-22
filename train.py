import shutil
import glob
import urllib.request
import tarfile
import argparse
import os


ap = argparse.ArgumentParser()
ap.add_argument("-d", "--data_path", type=str, required=True, help="path to dataset")
ap.add_argument("-m", "--object_detection_path", type=str, required=True, help="path to object detection")
ap.add_argument("-t", "--is_train", type=bool, default=False, help="whether need to train again")
ap.add_argument("-n", "--num_train_steps", type=int, default=5000, help="number of training steps")
ap.add_argument("-e", "--num_eval_steps", type=int, default=100, help="number of evaluation steps")
args = vars(ap.parse_args())


ROOT_PATH = os.getcwd()

DATASET_PATH = args["data_path"]
num_steps = args["num_train_steps"]  # number of training steps
num_eval_steps = args["num_eval_steps"]  # number of evaluation steps

# for training. Generally a higher batch size increases convergence of the model,
# but requires higher graphic memory.
MODELS_CONFIG = {
    "faster_rcnn_resnet50_v1": {
        "model_name": "faster_rcnn_resnet50_v1_640x640_coco17_tpu-8",
        "pipeline_file": "faster_rcnn_resnet50_v1_640x640_coco17_tpu-8.config",
        "batch_size": 32        
    },
    "efficientdet_d0": {
        "model_name": "efficientdet_d0_coco17_tpu-32",
        "pipeline_file": "ssd_efficientdet_d0_512x512_coco17_tpu-8.config",
        "batch_size": 8   # set 64 batch_size on two 2080ti, which summed up round 22g gpu memory, still OOM
    }
}

selected_model = 'efficientdet_d0'

MODEL_NAME = MODELS_CONFIG[selected_model]['model_name']
pipeline_file = MODELS_CONFIG[selected_model]['pipeline_file']
batch_size = MODELS_CONFIG[selected_model]['batch_size']

# where we save all downloaded pre-trained models
pretrained_download_models_path = os.path.join(ROOT_PATH, "downloaded_finetune_models") 
if not os.path.exists(pretrained_download_models_path):
    os.mkdir(pretrained_download_models_path)

# where we save all trained models on our custom datasets
custom_trained_models_path = os.path.join(ROOT_PATH, "custom_trained_models")
if not os.path.exists(custom_trained_models_path):
    os.mkdir(custom_trained_models_path)

# where we save designated pretrained model used for training
designated_trained_model_path = os.path.join(custom_trained_models_path, MODEL_NAME)
if not os.path.exists(designated_trained_model_path):
    os.mkdir(designated_trained_model_path)

DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/'
designated_pretrained_model_tar_path = os.path.join(pretrained_download_models_path, MODEL_NAME+".tar.gz")
if not os.path.exists(designated_pretrained_model_tar_path):
    urllib.request.urlretrieve(DOWNLOAD_BASE + MODEL_NAME + ".tar.gz", designated_pretrained_model_tar_path)
    print("[INFO] download pretrained model from => {}".format(designated_pretrained_model_tar_path))
else:
    print("[INFO] pretrained model existed in path => {}".format(designated_pretrained_model_tar_path))

if not os.path.join(os.path.join(pretrained_download_models_path, MODEL_NAME)):
    tar = tarfile.open(designated_pretrained_model_tar_path)
    tar.extractall(path=pretrained_download_models_path)
    tar.close()
    print("[INFO] extraced pretrained model to path => {}".foramt(pretrained_download_models_path))
else:
    print("[INFO] no need to extract .tar.gz file, model existed in path => {}".format(os.path.join(pretrained_download_models_path, MODEL_NAME)))

# copy tfod model config samples to our working space
pipeline_fname = os.path.join(os.getcwd(), pipeline_file)
shutil.copy(os.path.join(args["object_detection_path"], "configs", "tf2", pipeline_file), pipeline_fname)
print("[INFO] copy {} from {} to {}".format(pipeline_file, os.path.join(args["object_detection_path"], "object_detection", "configs", "tf2"), os.getcwd()))

# initialize paths required for training
test_record_fname = os.path.join(args["data_path"], 'test.record')
train_record_fname = os.path.join(args["data_path"],'train.record')
labelmap_pbtxt_fname = os.path.join(args["data_path"], 'labelmap.pbtxt')
finetune_checkpoint = os.path.join(pretrained_download_models_path, MODEL_NAME, "checkpoint/ckpt-0")

import re
import sys
sys.path.append(args["object_detection_path"])

from object_detection.utils import label_map_util

label_map = label_map_util.load_labelmap(labelmap_pbtxt_fname)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=90, use_display_name=True)
category_index = label_map_util.create_category_index(categories)
num_classes = len(category_index.keys())

with open(pipeline_fname) as f:
    s = f.read()

with open(pipeline_fname, 'w') as f:
    
    # fine_tune_checkpoint
    s = re.sub('fine_tune_checkpoint: ".*?"', 'fine_tune_checkpoint: "{}"'.format(finetune_checkpoint), s)
    
    # tfrecord files train and test.
    s = re.sub('(input_path: ".*?)(train2017)(.*?")', 'input_path: "{}"'.format(train_record_fname), s)
    s = re.sub('(input_path: ".*?)(val2017)(.*?")', 'input_path: "{}"'.format(test_record_fname), s)

    # label_map_path
    s = re.sub('label_map_path: ".*?"', 'label_map_path: "{}"'.format(labelmap_pbtxt_fname), s)

    # Set training batch_size.
    s = re.sub('batch_size: [0-9]+', 'batch_size: {}'.format(batch_size), s)

    # Set training steps, num_steps
    s = re.sub('num_steps: [0-9]+', 'num_steps: {}'.format(num_steps), s)
    
    # Set number of classes num_classes.
    s = re.sub('num_classes: [0-9]+','num_classes: {}'.format(num_classes), s)
    
    # Fine-tune checkpoint type
    s = re.sub('fine_tune_checkpoint_type: "classification"', 'fine_tune_checkpoint_type: "{}"'.format('detection'), s)
    
    s = re.sub('num_epochs: [0-9]+', 'num_epochs: {}'.format(num_eval_steps), s)

    f.write(s)

print("[INFO] modified config file based on our custom parameters")

if args["is_train"]:
    # start training
    os.system("python object_detection/model_main_tf2.py --pipeline_config_path={} --model_dir={} --alsologtostderr --num_train_steps={} --sample_1_of_n_eval_examples={} --num_eval_steps={}".
        format(pipeline_fname, os.path.join(custom_trained_models_path, MODEL_NAME), num_steps, 1, num_eval_steps))
    print("[INFO] training done")

# export model's final checkpoint 
trained_checkpoint_dir = os.path.join(custom_trained_models_path, MODEL_NAME)
output_directory = os.path.join(trained_checkpoint_dir, "export")
os.system("python object_detection/exporter_main_v2.py --trained_checkpoint_dir={} --output_directory={} --pipeline_config_path={}"
    .format(trained_checkpoint_dir, output_directory, pipeline_fname))
print("[INFO] export custom trained model's final checkpoint to => {}".format(output_directory))
