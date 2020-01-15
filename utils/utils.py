import csv
import numpy
import random
import nltk
import pandas as pd
import json

def take_key(ele):
    return int(ele[0])


def split_dataset(data_path: str = None, 
                  split_ratio: float = 0.8) -> None:
    with open(data_path, 'r') as data_file:
        csv_reader = csv.reader(data_file)
        head = next(csv_reader)
        data_list = []
        for line in csv_reader:
            data_list.append(line)
    random.shuffle(data_list)
    spilt_index = int(split_ratio*len(data_list))
    train_list = data_list[:spilt_index]
    val_list = data_list[spilt_index:]
    train_list.sort(key=take_key)
    val_list.sort(key=take_key)
    print('split_ratio:{}, all:{}, train:{}, val:{}'.format(split_ratio, len(data_list), len(train_list), len(val_list)))
    with open(data_path.replace('.csv', '_t.csv'), 'w') as train_f:
        csv_writer = csv.writer(train_f)
        csv_writer.writerow(head)
        for line in train_list:
            csv_writer.writerow(line)
    with open(data_path.replace('.csv', '_v.csv'), 'w') as val_f:
        csv_writer = csv.writer(val_f)
        csv_writer.writerow(head)
        for line in val_list:
            csv_writer.writerow(line)

def json_result2csv_submission(json_path: str = None) -> None:
    with open(json_path, 'r') as f:
        data_id_list = ['id']
        target_list = ['target']
        for line in f:
            loaded_dict = json.loads(line)
            data_id_list.append(loaded_dict['data_id'][0])
            target_list.append(loaded_dict['label'])
    with open(json_path.replace('.json', '.csv'), 'w') as result_f:
        csv_writer = csv.writer(result_f)
        for data_id, target in zip(data_id_list, target_list):
            # print(data_id, target)
            csv_writer.writerow([data_id, target])

def data_conbine(data1_path, data2_path):
    with open(data1_path, 'r') as data1_file:
        data1_reader = csv.reader(data1_file)
        head = next(data1_reader)
    with open(data2_path, 'r') as data2_file:
        data2_reader = csv.reader(data2_file)
        head = next(data2_reader)
    conbined_data_list = []
    for line in data1_reader:
        conbined_data_list.append(line)
    for line in data2_reader:
        conbined_data_list.append(line)
    conbined_data_list.sort(key=take_key)
    with open('./data/conbined_data.csv', 'w') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(head)
        for line in conbined_data_list:
            csv_writer.writerow(line)

def txt2csv(ori_path, aug_path):
    ori_list = []
    aug_list = []
    with open(ori_path, 'r') as ori_file:
        ori_reader = csv.reader(ori_file)
        head = next(ori_reader)
        for line in ori_reader:
            ori_list.append(line)
    with open(aug_path, 'r') as aug_file:
        for line in aug_file:
            aug_list.append(line.strip()[2:])
        print(len(ori_list), '#'*3, len(aug_list))
    new_aug_list = []
    for index, ori_line in enumerate(ori_list):
        for i in range(0, 2):
            ori_line[3] = aug_list[index*2+i]
            new_aug_list.append(list(ori_line))
    with open(aug_path.replace('.txt', '.csv'), 'w') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(head)
        for line in new_aug_list:
            csv_writer.writerow(line)

def csv2txt(csv_path):
    ori_list = []
    with open(csv_path, 'r') as ori_file:
        ori_reader = csv.reader(ori_file)
        head = next(ori_reader)
        for line in ori_reader:
            ori_list.append(line)
    txt_path = csv_path.replace('.csv', '.txt')
    with open(txt_path, 'w') as txt_file:
        for line in ori_list:
            label = line[4]
            text = line[3].replace('\n', '')
            txt_file.writelines([label, '\t', text, '\n'])


if __name__ == "__main__":
    # split_dataset('./data/train.csv')
    # json_result2csv_submission('./DT_predict.json')
    txt2csv('/Users/jinlukang/Desktop/JD/NLP/Disaster_Tweets/data/train_t.csv', '/Users/jinlukang/Desktop/JD/NLP/Disaster_Tweets/data/eda_train_t.txt')
    # csv2txt('./data/train_t.csv')