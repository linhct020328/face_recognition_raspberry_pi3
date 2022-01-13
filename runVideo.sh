#!/bin/bash

python pi_video_file.py --input videos/${@} --cascade haarcascade_frontalface_default.xml --encodings encodings.pickle
