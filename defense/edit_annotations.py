import json 
import os
import copy



train_dict = json.load(open('train2017_orig.json'))
new_list = []
print(train_dict['images'][2])
print(train_dict['annotations'][2])

for imageidx in range(len(train_dict['images'])):
       
    new_list.append(copy.deepcopy(train_dict['images'][imageidx]))
    new_list[-1]['file_name'] = new_list[-1]["file_name"].split(".")[0] + ".1." + new_list[-1]["file_name"].split(".")[1]


train_dict['images'].extend(new_list)
train_dict['annotations'] += train_dict['annotations']
print(len(train_dict['annotations']))

json.dump(train_dict,open('train2017.json',mode="w"))
print(len(json.load(open('train2017.json'))['images']))

test_dict = json.load(open('test2017_orig.json'))
new_list = []    

for imageidx in range(len(test_dict['images'])):
    new_list.append(copy.deepcopy(test_dict['images'][imageidx]))
    new_list[-1]['file_name'] = new_list[-1]["file_name"].split(".")[0] + ".1." + new_list[-1]["file_name"].split(".")[1]

test_dict['images'] += new_list
test_dict['annotations'] += test_dict['annotations']

json.dump(test_dict,open('test2017.json',mode="w"))

val_dict = json.load(open('val2017_orig.json'))
new_list = []    

for imageidx in range(len(val_dict['images'])):
    new_list.append(copy.deepcopy(val_dict['images'][imageidx]))
    new_list[-1]['file_name'] = new_list[-1]["file_name"].split(".")[0] + ".1." + new_list[-1]["file_name"].split(".")[1]


val_dict['images'] += new_list
val_dict['annotations'] += val_dict['annotations']

json.dump(val_dict,open('val2017.json',mode="w"))


