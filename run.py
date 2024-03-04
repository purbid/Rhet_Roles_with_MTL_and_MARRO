import os
import sys
import sys
import logging
import argparse

from pathlib import Path 
from logger_file import fetch_logger
from os.path import join
from pathlib import Path
from datetime import datetime

now = datetime.now()
dt_string = now.strftime("%d-%m-%Y-%H:%M:%S")
# logger_filename = '/content/drive/MyDrive/IIT_law_ai/semantic_segmentation/experiment_logs/copied_weights/'+str(dt_string)+'test_logger.log'
# if not os.path.exists(logger_filename):
#   Path(logger_filename).touch()
# logging.basicConfig(filename=logger_filename, filemode='w', format='%(name)s - %(levelname)s - %(message)s')

# logger = fetch_logger('run file logger')
# logger.info("Starting the run for Rhetoric role classification")
# logger.info("test statement")


from model.Hier_BiLSTM_CRF import *
from model.Lstm_attn_CRF import *
from model.bert_CRF import *
from model.Lstm_attn import *
from statistics import mean
import pandas as pd
from model.tf_Hier_BiLSTM_CRF import *
from model.tf_atten_BiLSTM_CRF import *
from model.attn_crf import *
from prepare_data import *
from train import *

import torch

import random
import numpy as np

SEED = 42

np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

import warnings

warnings.filterwarnings("ignore", category=Warning)


def main(root = "", country = "UK"):
    parser = argparse.ArgumentParser()

    parser.add_argument('--pretrained', default=False, type=bool,
                        help='Whether the model uses pretrained sentence embeddings or not')



    ####### data related args ##############
    parser.add_argument('--country', default='IN')
    parser.add_argument('--data_path', default='{}/dataset/{}-train-set/'.format(root, country),
                        type=str, help='Folder to store the annotated text files')
    parser.add_argument('--data_docs_original',
                        default='{}/dataset/{}-train-set/'.format(root, country),
                        type=str, help='original sentences and files')
    parser.add_argument('--save_path',
                        default='{}/saved/saved_{}_new/'.format(root, country), type=str,
                        help='Folder where predictions and models will be saved')
    parser.add_argument('--cat_path',
                        default='{}/{}_categories.txt'.format(root, country.lower()), type=str,
                        help='Path to file containing category details')

    parser.add_argument('--dataset_size', default=50, type=int, help='Total no. of docs')
    parser.add_argument('--num_folds', default=5, type=int, help='No. of folds to divide the dataset into')
    parser.add_argument('--device', default='cpu', type=str, help='cuda / cpu')
    parser.add_argument('--use_tf_emb_plus_attention', default=False, type=bool, help='use the tf embeddigns instead of the current sent2vec embeddings, add attention')
    parser.add_argument('--use_tf_lstm_crf', default=False, type=bool)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--print_every', default=1, type=int, help='Epoch interval after which validation macro f1 and loss will be printed')
    parser.add_argument('--lr', default=0.001, type=float, help='Learning Rate')
    parser.add_argument('--min_lr', default=5 / 10000, type=float, help='min Learning Rate')
    parser.add_argument('--lr_decay_factor', default=0.95, type=float, help='Learning Rate Decay')
    parser.add_argument('--reg', default=0, type=float, help='L2 Regularization')
    parser.add_argument('--emb_dim', default=512, type=int, help='Sentence embedding dimension')

    parser.add_argument('--word_emb_dim', default= 200, type=int, help='Word embedding dimension, applicable only if pretrained = False')
    parser.add_argument('--epochs', default = 50, type=int)
    parser.add_argument('--val_fold', default='cross', type=str, help='Fold number to be used as validation, use cross for num_folds cross validation')
    parser.add_argument('--use_attention', default=False, type=bool)
    parser.add_argument('--attention_heads', default=5, type=int)
    parser.add_argument('--encoder_blocks', default=2, type=int)

    args = parser.parse_args()

    ##############################

    dir_name = 'experiment_logs/{}-{}-{}-{}-{}/'.format(country, args.batch_size, args.attention_heads, args.encoder_blocks, args.epochs)
    
    log_dir = join(root, dir_name)

    if not Path(log_dir).exists():
          Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    logger_filename = str(dt_string)+'test_logger.log'
    logger_filename_full = join(log_dir, logger_filename)
    
    if not os.path.exists(logger_filename_full):
      Path(logger_filename_full).touch()
    
    logging.basicConfig(filename=logger_filename_full, filemode='w', format='%(name)s - %(levelname)s - %(message)s')

    logger = fetch_logger('run file logger')
    logger.info("Starting the run for Rhetoric role classification")
    logger.info("test statement")

    ##########################
    
    logger.info('model is for : {}'.format(args.cat_path))
    logger.info('model is pretrained : {}'.format(args.pretrained))
    logger.info('using attention model : {}'.format(args.use_attention))
    logger.info('usinf tf embedding plus attention model : {}'.format(args.use_tf_emb_plus_attention))
    logger.info('attention heads used in model : {}'.format(args.attention_heads))
    logger.info('encoder blocks for attention : {}'.format(args.encoder_blocks))
    logger.info('training batch size : {}'.format(args.batch_size))
    logger.info('training learning rate : {}'.format(args.lr))
    logger.info('training epochs : {}'.format(args.epochs))

    print('\nPreparing data ...', end=' ')


    idx_order = prepare_folds(args)
    x, y, word2idx, tag2idx, original_sentences = prepare_data_original(idx_order, args)

    print('Done')
    print(len(x), len(y))

    print('Vocabulary size:', len(word2idx))
    print('#Tags:', len(tag2idx))




    # Dump word2idx and tag2idx
    with open(args.save_path + 'word2idx.json', 'w') as fp:
        json.dump(word2idx, fp)
    with open(args.save_path + 'tag2idx.json', 'w') as fp:
        json.dump(tag2idx, fp)

    if args.val_fold == 'cross':

        print('\nCross-validation\n')
        avg_f1 = []
        attention_df = []
        for f in range(args.num_folds):
            if f!=0:
               continue
            print('\nInitializing model ...', end=' ')

            if args.use_attention == True:
                model = Attn_BiLSTM_CRF(len(tag2idx), args.emb_dim, tag2idx['<start>'], tag2idx['<end>'],
                                        tag2idx['<pad>'], vocab_size=len(word2idx), word_emb_dim=args.word_emb_dim,
                                        pretrained=args.pretrained, device=args.device).to(args.device)

            elif args.use_tf_lstm_crf == True:

                print("before making model")
                model = Tf_Hier_LSTM_CRF_Classifier(len(tag2idx), args.emb_dim, tag2idx['<start>'], tag2idx['<end>'],
                                 tag2idx['<pad>'], vocab_size=len(word2idx), word_emb_dim=args.word_emb_dim,
                                 pretrained=args.pretrained, device=args.device).to(args.device)

            elif args.use_tf_emb_plus_attention == True:
                model = Tf_Attn_Hier_LSTM_CRF_Classifier(len(tag2idx), args.emb_dim, tag2idx['<start>'],
                                                         tag2idx['<end>'], tag2idx['<pad>'], vocab_size=len(word2idx),
                                                         word_emb_dim=args.word_emb_dim, pretrained=args.pretrained,
                                                         device=args.device, attention_heads=args.attention_heads,
                                                         num_blocks=args.encoder_blocks).to(args.device)

            else:
                print("here taking old model")
                model = Hier_LSTM_CRF_Classifier(len(tag2idx), args.emb_dim, tag2idx['<start>'], tag2idx['<end>'],
                                                 tag2idx['<pad>'], vocab_size=len(word2idx),
                                                 word_emb_dim=args.word_emb_dim, pretrained=args.pretrained,
                                                 device=args.device).to(args.device)

            print('Done')

            print('\nEvaluating on fold', f, '...')
            if f == 0 :
              logger.info("the model architecture is : \n {}".format(model))
            print(len(x), len(y), idx_order)
            curr_fold_f1, attention_df_curr = learn(model, x, y, tag2idx, f, args, idx_order)

            avg_f1.append(curr_fold_f1)
            attention_df.extend(attention_df_curr.copy())

        print("average F1 across folds is : " + str(mean(avg_f1)))


if __name__ == '__main__':
    root  = os.getcwd()
    # root = os.path.join(root, 'drive/MyDrive/IIT_law_ai/semantic_segmentation')
    main(root, "UK")
