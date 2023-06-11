import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# Very important very obscure flags, make things go brrrr
# They incur high startup times though, so only useful for big jobs
os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit"

# Reduce logging
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

import tensorflow as tf

from Generator.TFDataReader import DataGenerator

# Stop TF from hogging all of the VRAM
gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


@tf.function
def pred_model(x):
    return model(x, training=False)


def process_images(filepaths, images):
    preds = pred_model(images).numpy()

    for image_path, pred in zip(filepaths, preds):
        image_path = image_path.numpy().decode("utf-8")

        tags_df["preds"] = pred
        general_tags = tags_df[tags_df["category"] == 0]
        chosen_tags = general_tags[general_tags["preds"] > threshold]
        chosen_tags = chosen_tags.sort_values(by="preds", ascending=False)
        tags_names = chosen_tags["sanitized_name"]
        tags_string = ", ".join(tags_names)
        with open(Path(image_path).with_suffix(".txt"), "w") as f:
            f.write(tags_string)


def dataset_diagnostic(filepaths, images):
    lines = []
    for image_path in filepaths:
        image_path = image_path.numpy().decode("utf-8")
        lines.append(f"{image_path}\n")
    with open("dry_run_read.txt", "a") as f:
        f.writelines(lines)


parser = argparse.ArgumentParser(description="Mass tag a set of images")

# Images arguments
parser.add_argument(
    "--targets-path",
    required=True,
    help="Folder with the images to tag",
)
parser.add_argument(
    "--recursive",
    action="store_true",
    help="Recurse directories when looking for images",
)
parser.set_defaults(recursive=False)
parser.add_argument(
    "--dry-run",
    action="store_true",
    help="Load the images without running predictions",
)
parser.set_defaults(dry_run=False)

# Model arguments
parser.add_argument("--model-folder", default="networks/wd-v1-4-moat-tagger-v2")
parser.add_argument("--tags-csv", default="selected_tags.csv")
parser.add_argument(
    "--threshold",
    default=0.35,
    type=float,
    help="Predictions threshold",
)
parser.add_argument(
    "--batch-size",
    default=32,
    type=int,
    help="Batch size",
)
args = parser.parse_args()

targets_path = args.targets_path
glob_pattern = "**/*" if args.recursive else "*"
dry_run = args.dry_run

model_folder = args.model_folder
labels_file = args.tags_csv
threshold = args.threshold
batch_size = args.batch_size

image_extensions = [".jpeg", ".jpg", ".png", ".webp"]

# https://github.com/toriato/stable-diffusion-webui-wd14-tagger/blob/a9eacb1eff904552d3012babfa28b57e1d3e295c/tagger/ui.py#L368
kaomojis = [
    "0_0",
    "(o)_(o)",
    "+_+",
    "+_-",
    "._.",
    "<o>_<o>",
    "<|>_<|>",
    "=_=",
    ">_<",
    "3_3",
    "6_9",
    ">_o",
    "@_@",
    "^_^",
    "o_o",
    "u_u",
    "x_x",
    "|_|",
    "||_||",
]

tags_df = pd.read_csv(labels_file)
tags_df["sanitized_name"] = tags_df["name"].map(
    lambda x: x.replace("_", " ") if x not in kaomojis else x
)

images_list = list(
    (
        str(p.resolve())
        for p in Path(targets_path).glob(glob_pattern)
        if p.suffix.lower() in image_extensions
    )
)

if not dry_run:
    model = tf.keras.models.load_model(model_folder)
    _, height, width, _ = model.inputs[0].shape
    process_func = process_images
else:
    height, width = 224, 224
    process_func = dataset_diagnostic

    scheduled = [f"{image_path}\n" for image_path in images_list]

    # Truncate the file from previous runs
    open("dry_run_read.txt", "w").close()
    with open("dry_run_scheduled.txt", "w") as f:
        f.writelines(scheduled)


generator = DataGenerator(
    file_list=images_list, target_size=height, batch_size=batch_size
).genDS()

for filepaths, images in tqdm(generator):
    process_func(filepaths, images)
