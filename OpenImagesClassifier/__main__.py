from OpenImagesClassifier import user_interface
import os
import argparse

parser = argparse.ArgumentParser(description='Process run mode')
parser.add_argument('-chdir', action='store_true')
args = parser.parse_args()

if args.chdir:
    os.chdir('./OpenImagesClassifier')

user_interface.main()