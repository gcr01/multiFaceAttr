import os
import shutil
import argparse
import csv

ap=argparse.ArgumentParser()
ap.add_argument('--src',required=True,
                help='Path of source image')
ap.add_argument('--dst',required=True,
                help='Path of target folder')
args = vars(ap.parse_args())

def copy_img(src_path,dst_path):
    with open('label-imdb-age.csv','r') as csvfile:
        reader = csv.reader(csvfile,delimiter = ',')
        for line in reader:
            name = line[0]
            src = os.path.join(src_path,name)
            shutil.copy(src,dst_path)
if __name__=='__main__':
    copy_img(args['src'],args['dst'])
