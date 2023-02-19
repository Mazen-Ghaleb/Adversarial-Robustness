#! /usr/bin/bash
echo "Setting up YOLOX"

git submodule init
git submodule update

cd model/YOLOX
pip install -e .
cd ../..

echo "Setting up local packages"
pip install -e .