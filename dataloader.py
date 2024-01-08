from torch.utils.data import DataLoader
from utils import *


import os
import shutil
from sklearn.model_selection import train_test_split

def get_files(input_dir, test_size=0.2, random_state=42):
    files = os.listdir(input_dir)

    train_files, test_files = train_test_split(files, test_size=test_size, random_state=random_state)


    train_files_A, train_files_B = train_test_split(train_files, test_size=0.5, random_state=random_state)
    train_output_directory_A = os.path.join(input_dir, 'train_A')
    train_output_directory_B = os.path.join(input_dir, 'train_B')
    test_output_directory_A = os.path.join(input_dir, 'test_A')
    test_output_directory_B = os.path.join(input_dir, 'test_B')
    for file in train_files_A:
        src_path = os.path.join(input_dir, file)
        dest_path = os.path.join(train_output_directory_A, file)
        shutil.copy(src_path, dest_path)

    for file in train_files_B:
        src_path = os.path.join(input_dir, file)
        dest_path = os.path.join(train_output_directory_B, file)
        shutil.copy(src_path, dest_path)

    for file in test_files:
        src_path = os.path.join(input_dir, file)
        dest_path = os.path.join(test_output_directory_A, file)
        shutil.copy(src_path, dest_path)

    for file in test_files:
        src_path = os.path.join(input_dir, file)
        dest_path = os.path.join(test_output_directory_B, file)
        shutil.copy(src_path, dest_path)

    return train_files_A, train_files_B, test_files, test_files

def get_train_test_data(config,train_A_files,train_B_files,test_B_files,train_attack,test_attack):

    if train_attack == 'your attack' and test_attack == 'your attack':
        train_A_data = attack(train_A_files, sigma=0.1)
        train_B_data = attack(train_B_files, sigma=0.1)

        test_B_data = attack(test_B_files, sigma=0.1)

    train_A_dataloader = DataLoader(train_A_data, batch_size=config['b_size'], shuffle=True, \
                                    num_workers=config['num_workers'], drop_last=True)
    train_B_dataloader = DataLoader(train_B_data, batch_size=config['b_size'], shuffle=True, \
                                    num_workers=config['num_workers'], drop_last=True)
    test_B_dataloader = DataLoader(test_B_data, batch_size=1, shuffle=True)
    
    test_B_data = []
    for i, audio_pair in enumerate(test_B_dataloader):
        if i >= TEST_SIZE: break
        test_B_data.append(audio_pair)
    return train_A_dataloader, train_B_dataloader, test_B_data