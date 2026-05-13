#!/bin/bash
# Install system dependencies
apt-get update
apt-get install -y libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1

# Upgrade pip
pip install --upgrade pip

# Install tensorflow separately (in case of issues)
pip install tensorflow==2.15.0
