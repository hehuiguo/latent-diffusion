from curses.ascii import isdigit
from genericpath import isfile
import os
import shutil
import fire

img_file_end = [".jpg", ".JPG", ".png", ".PNG", ".webp", ".WEBP"]

def copy_img(source_path, destination_path):
    for file in os.listdir(source_path):
        file_path = os.path.join(source_path, file)
        if os.path.isfile(file_path):
            for x in img_file_end:
                if file.endswith(x):
                    shutil.copyfile(file_path, os.path.join(destination_path, file))
                else:
                    continue
        elif os.path.isdir(file_path):
            copy_img(file_path, destination_path)
        else:
            print(f"should not happend: {file_path}")


if __name__ == "__main__":
    fire.Fire()

# examples:
# python ./mv_lsun_data.py copy_img --source_path /data/huiguohe/datasets/LSUN/images/bedroom_train/ --destination_path ./data/lsun/bedrooms
