#!/bin/bash

# Upgrade pip
pip install --upgrade pip

pip install streamlit-webrtc==0.45.1

# Install other dependencies
pip install -r requirements.txt
