# Model Installation
To install the model use the following commands 

```bash
git clone git@github.com:Megvii-BaseDetection/YOLOX.git
cd YOLOX
pip3 install -v -e .  # or  python3 setup.py develop
cd ..
```

# Data Preparation
To put the data into coco format required by the yolox architecture do the following:
1. put all the files from tsinghua and gtsdb into datasets/tsinghua_gtsdb_full
2. make sure that there is not subdirectories in that directory 
3. run the following command 
```bash 
python3 prepare-data.py --root datasets/tsinghua_gtsdb_full
```

This will split the data into 70% train, 15% test, and 15% validation in datasets/tsinghua_gtsdb_speedlimit

# Model Training
To start training do the following 

1. Write an exp class simmilar to speedlimit_exp.py 
2. Download the pretrained weights for <a href="https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.pth">yolox-s</a>
3. Run the following command

```bash
python3 YOLOX/tools/train.py -f speedlimit_exp.py -d 1 -b 64 --fp16 -o -c yolox_s.pth --cache
```

for more ifno about training check <a href="https://github.com/Megvii-BaseDetection/YOLOX/blob/main/docs/train_custom_data.md">Training on custom data</a>

# Model Evaluation
1. Write an exp class simmilar to speedlimit_exp.py 
2. Run the following command

```bash
python3 YOLOX/tools/eval.py -f speedlimit_exp.py -c best_ckpt.pth -b 64 -d 1 --conf 0.1 --test --fp16 --fuse 
```