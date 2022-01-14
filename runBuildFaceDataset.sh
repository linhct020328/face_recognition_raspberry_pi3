#!/bin/bash

python build_face_dataset.py --cascade haarcascade_frontalface_default.xml --output dataset/${@}
