from model import *
from dataloader import *
from metrics import *
from utils import *

import os
import argparse
import yaml
import pickle


# Command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--device_id', type=int, default=0)
parser.add_argument('--config_file',type=str, default='config.yaml')
parser.add_argument('--pretrain_clean', type=bool, default=False)
parser.add_argument('--pretrain_CAE', type=str, default='cae.pkl')
parser.add_argument('--pretrain', type=bool, default=False)
parser.add_argument('--pretrain_AAE', type=str)
args = parser.parse_args()

BASE_DIR = '.'

# Select device
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device_id)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Get configuration
config = get_config(args.config_file)
config['attack'] = args.attack

train_A_files, train_B_files, test_A_files, test_B_files = get_files(input_dir='your dir')
train_attack = 'fgsm'
test_attack = 'SSAH'
train_A_dataloader, train_B_dataloader, test_B_data = get_train_test_data(config,train_A_files,train_B_files,test_B_files,train_attack,test_attack)

if args.pretrain:
    with open(args.pretrain_AAE, 'rb') as input:
        trainer = pickle.load(input)
elif args.pretrain_clean:
    with open(args.pretrain_CAE, 'rb') as input:
        vae_clean = pickle.load(input)
        trainer = SSE(config,vae_clean).to(device)
else:
    trainer = SSE(config).to(device)


if not args.pretrain_clean:

    for epoch in range(1,config['epochs_a']+1):

        for i, audio in enumerate(train_A_dataloader):
            loss, x_a_recon = trainer(audio.float().to(device),'a')


    with open(args.pretrain_CAE, 'wb') as output:
        pickle.dump(trainer.gen_a, output, pickle.HIGHEST_PROTOCOL)


for epoch in range(1,config['epochs_b']+1):
    for i, (audio_b,audio_b_clean) in enumerate(train_B_dataloader):
        loss,x_b_recon,x_ba = trainer(audio_b.float().to(device),'b')

        if i == len(train_B_dataloader)-1 and epoch%10 == 0:

            scores = 0

            for i, (b_test,label) in enumerate(test_B_data):
                predict = trainer(audio_b_test.float().to(device),'eval')
                score = dr(predict, label)
                scores.append(score)

            # Save model periodically
            if epoch%100 == 0:
                with open('{}/sep_trainer_ep{}.pkl'.format(BASE_DIR,epoch), 'wb') as output:
                    pickle.dump(trainer, output, pickle.HIGHEST_PROTOCOL)
                avg_score = mean(scores)

with open('{}/sep_trainer_final.pkl'.format(BASE_DIR), 'wb') as output:
    pickle.dump(trainer, output, pickle.HIGHEST_PROTOCOL)
