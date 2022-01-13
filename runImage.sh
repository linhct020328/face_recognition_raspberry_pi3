#!/bin/bash

python pi_image.py --image images/${@} --cascade haarcascade_frontalface_default.xml --encodings encodings.pickle
