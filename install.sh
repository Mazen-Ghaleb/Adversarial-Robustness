#! /usr/bin/bash
echo "Setting up YOLOX"
cd model/YOLOX
pip install -e .
cd ../..

echo "Setting up local packages"
pip install -e .