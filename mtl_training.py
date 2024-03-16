
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_sequence, pack_padded_sequence, pad_packed_sequence
import time
import json
import random
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
import string
from collections import defaultdict
import os

SEED = 42
import itertools


torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

'''
Shift Module:
    A Bi-LSTM is used to generate feature vectors for each sentence from the sentence embeddings.
    The feature vectors are actually context-aware sentence embeddings.
    These are then fed to a feed-forward network to obtain emission scores for each class at each sentence.
'''
class LSTM_Emitter_Binary(nn.Module):
    def __init__(self, n_tags, emb_dim, hidden_dim, drop = 0.5, device = 'cuda'):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(512, 1024)
        self.fc2 = nn.Linear(1024, 512)

        self.lstm = nn.LSTM(emb_dim, hidden_dim // 2, bidirectional = True, batch_first = True)
        self.dropout = nn.Dropout(drop)
        self.hidden2tag = nn.Linear(hidden_dim, n_tags)
        self.hidden = None
        self.device = device

    def init_hidden(self, batch_size):
        return (torch.randn(2, batch_size, self.hidden_dim // 2).to(self.device), torch.randn(2, batch_size, self.hidden_dim // 2).to(self.device))

    def forward(self, sequences, sequences_rhet):
        ## sequences: tensor[batch_size, max_seq_len, emb_dim]

        sequences = self.fc1(sequences).to(self.device)
        sequences = self.fc2(sequences).to(self.device)

        sequences = torch.cat((sequences, sequences_rhet), 2).to(self.device)

        self.hidden = self.init_hidden(sequences.shape[0])


        x, self.hidden = self.lstm(sequences, self.hidden)
        x_new = self.dropout(x)

        x_new = self.hidden2tag(x_new)
        return x_new, x

'''
RR Module:
    A Bi-LSTM is used to generate feature vectors for each sentence from the sentence embeddings.
    The feature vectors are actually context-aware sentence embeddings.
    These are then fed to a feed-forward network to obtain emission scores for each class at each sentence.
'''
class LSTM_Emitter(nn.Module):
    def __init__(self, n_tags, emb_dim, hidden_dim, drop = 0.5, device = 'cuda'):
        super().__init__()

        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(emb_dim, hidden_dim // 2, bidirectional = True, batch_first = True)
        self.dropout = nn.Dropout(drop)
        self.hidden2tag = nn.Linear(2*hidden_dim, n_tags)
        self.hidden = None
        self.device = device

    def init_hidden(self, batch_size):
        return (torch.randn(2, batch_size, self.hidden_dim // 2).to(self.device), torch.randn(2, batch_size, self.hidden_dim // 2).to(self.device))

    def forward(self, sequences, hidden_binary):

        self.hidden = self.init_hidden(sequences.shape[0])

        # generate context-aware sentence embeddings (feature vectors)
        x, self.hidden = self.lstm(sequences, self.hidden)
        final = torch.zeros((x.shape[0], x.shape[1], 2*x.shape[2])).to(self.device)
        for batch_name, doc in enumerate(x):
            for i, sent in enumerate(doc):
                final[batch_name][i] = torch.cat((x[batch_name][i], hidden_binary[batch_name][i]),0)
        final = self.dropout(final)

        final = self.hidden2tag(final)
        return final

'''
    A linear-chain CRF is fed with the emission scores at each sentence,
    and it finds out the optimal sequence of tags by learning the transition scores.
'''
class CRF(nn.Module):
    def __init__(self, n_tags, sos_tag_idx, eos_tag_idx, pad_tag_idx = None):
        super().__init__()

        self.n_tags = n_tags
        self.SOS_TAG_IDX = sos_tag_idx
        self.EOS_TAG_IDX = eos_tag_idx
        self.PAD_TAG_IDX = pad_tag_idx

        self.transitions = nn.Parameter(torch.empty(self.n_tags, self.n_tags))
        self.init_weights()

    def init_weights(self):
        # initialize transitions from random uniform distribution between -0.1 and 0.1
        nn.init.uniform_(self.transitions, -0.1, 0.1)

        # enforce constraints (rows = from, cols = to) with a big negative number.
        # exp(-1000000) ~ 0

        # no transitions to SOS
        self.transitions.data[:, self.SOS_TAG_IDX] = -1000000.0
        # no transition from EOS
        self.transitions.data[self.EOS_TAG_IDX, :] = -1000000.0

        if self.PAD_TAG_IDX is not None:
            # no transitions from pad except to pad
            self.transitions.data[self.PAD_TAG_IDX, :] = -1000000.0
            self.transitions.data[:, self.PAD_TAG_IDX] = -1000000.0
            # transitions allowed from end and pad to pad
            self.transitions.data[self.PAD_TAG_IDX, self.EOS_TAG_IDX] = 0.0
            self.transitions.data[self.PAD_TAG_IDX, self.PAD_TAG_IDX] = 0.0

    def forward(self, emissions, tags, mask = None):
        ## emissions: tensor[batch_size, seq_len, n_tags]
        ## tags: tensor[batch_size, seq_len]
        ## mask: tensor[batch_size, seq_len], indicates valid positions (0 for pad)
        return -self.log_likelihood(emissions, tags, mask = mask)

    def log_likelihood(self, emissions, tags, mask = None):
        if mask is None:
            mask = torch.ones(emissions.shape[:2])

        scores = self._compute_scores(emissions, tags, mask = mask)
        partition = self._compute_log_partition(emissions, mask = mask)
        return torch.sum(scores - partition)

    # find out the optimal tag sequence using Viterbi Decoding Algorithm
    def decode(self, emissions, mask = None):
        if mask is None:
            mask = torch.ones(emissions.shape[:2])

        scores, sequences = self._viterbi_decode(emissions, mask)
        return scores, sequences

    def _compute_scores(self, emissions, tags, mask):
        batch_size, seq_len = tags.shape
        if(torch.cuda.is_available()):
            scores = torch.zeros(batch_size).cuda()
        else:
            scores = torch.zeros(batch_size)

        # save first and last tags for later
        first_tags = tags[:, 0]
        last_valid_idx = mask.int().sum(1) - 1
        last_tags = tags.gather(1, last_valid_idx.unsqueeze(1)).squeeze()

        # add transition from SOS to first tags for each sample in batch
        t_scores = self.transitions[self.SOS_TAG_IDX, first_tags]

        # add emission scores of the first tag for each sample in batch
        e_scores = emissions[:, 0].gather(1, first_tags.unsqueeze(1)).squeeze()
        scores += e_scores + t_scores

        # repeat for every remaining word
        for i in range(1, seq_len):

            is_valid = mask[:, i]
            prev_tags = tags[:, i - 1]
            curr_tags = tags[:, i]

            e_scores = emissions[:, i].gather(1, curr_tags.unsqueeze(1)).squeeze()
            t_scores = self.transitions[prev_tags, curr_tags]

            # apply the mask
            e_scores = e_scores * is_valid
            t_scores = t_scores * is_valid

            scores += e_scores + t_scores

        # add transition from last tag to EOS for each sample in batch
        scores += self.transitions[last_tags, self.EOS_TAG_IDX]
        return scores

    # compute the partition function in log-space using forward algorithm
    def _compute_log_partition(self, emissions, mask):
        batch_size, seq_len, n_tags = emissions.shape

        # in the first step, SOS has all the scores
        alphas = self.transitions[self.SOS_TAG_IDX, :].unsqueeze(0) + emissions[:, 0]

        for i in range(1, seq_len):
            ## tensor[batch_size, n_tags] -> tensor[batch_size, 1, n_tags]
            e_scores = emissions[:, i].unsqueeze(1)

            ## tensor[n_tags, n_tags] -> tensor[batch_size, n_tags, n_tags]
            t_scores = self.transitions.unsqueeze(0)

            ## tensor[batch_size, n_tags] -> tensor[batch_size, n_tags, 1]
            a_scores = alphas.unsqueeze(2)

            scores = e_scores + t_scores + a_scores
            new_alphas = torch.logsumexp(scores, dim = 1)

            # set alphas if the mask is valid, else keep current values
            is_valid = mask[:, i].unsqueeze(-1)
            alphas = is_valid * new_alphas + (1 - is_valid) * alphas

        # add scores for final transition
        last_transition = self.transitions[:, self.EOS_TAG_IDX]
        end_scores = alphas + last_transition.unsqueeze(0)

        # return log_sum_exp
        return torch.logsumexp(end_scores, dim = 1)

    # return a list of optimal tag sequence for each example in the batch
    def _viterbi_decode(self, emissions, mask):
        batch_size, seq_len, n_tags = emissions.shape

        # in the first iteration, SOS will have all the scores and then, the max
        alphas = self.transitions[self.SOS_TAG_IDX, :].unsqueeze(0) + emissions[:, 0]

        backpointers = []

        for i in range(1, seq_len):
            ## tensor[batch_size, n_tags] -> tensor[batch_size, 1, n_tags]
            e_scores = emissions[:, i].unsqueeze(1)

            ## tensor[n_tags, n_tags] -> tensor[batch_size, n_tags, n_tags]
            t_scores = self.transitions.unsqueeze(0)

            ## tensor[batch_size, n_tags] -> tensor[batch_size, n_tags, 1]
            a_scores = alphas.unsqueeze(2)

            scores = e_scores + t_scores + a_scores

            # find the highest score and tag, instead of log_sum_exp
            max_scores, max_score_tags = torch.max(scores, dim = 1)

            # set alphas if the mask is valid, otherwise keep the current values
            is_valid = mask[:, i].unsqueeze(-1)
            alphas = is_valid * max_scores + (1 - is_valid) * alphas

            backpointers.append(max_score_tags.t())

        # add scores for final transition
        last_transition = self.transitions[:, self.EOS_TAG_IDX]
        end_scores = alphas + last_transition.unsqueeze(0)

        # get the final most probable score and the final most probable tag
        max_final_scores, max_final_tags = torch.max(end_scores, dim=1)

        # find the best sequence of labels for each sample in the batch
        best_sequences = []
        emission_lengths = mask.int().sum(dim=1)
        for i in range(batch_size):

            # recover the original sentence length for the i-th sample in the batch
            sample_length = emission_lengths[i].item()

            # recover the max tag for the last timestep
            sample_final_tag = max_final_tags[i].item()

            # limit the backpointers until the last but one
            # since the last corresponds to the sample_final_tag
            sample_backpointers = backpointers[: sample_length - 1]

            # follow the backpointers to build the sequence of labels
            sample_path = self._find_best_path(i, sample_final_tag, sample_backpointers)

            # add this path to the list of best sequences
            best_sequences.append(sample_path)

        return max_final_scores, best_sequences

    # auxiliary function to find the best path sequence for a specific example
    def _find_best_path(self, sample_id, best_tag, backpointers):
        ## backpointers: list[tensor[seq_len_i - 1, n_tags, batch_size]], seq_len_i is the length of the i-th sample of the batch

        # add the final best_tag to our best path
        best_path = [best_tag]

        # traverse the backpointers in backwards
        for backpointers_t in reversed(backpointers):

            # recover the best_tag at this timestep
            best_tag = backpointers_t[best_tag][sample_id].item()

            # append to the beginning of the list so we don't need to reverse it later
            best_path.insert(0, best_tag)

        return best_path

"""ATTENTION HELPER FUNCTIONS"""

from torch import nn
def attention_padding_mask(q, k, padding_index=0):
    """Generate mask tensor for padding value
    Args:
        q (Tensor): (B, T_q)
        k (Tensor): (B, T_k)
        padding_index (int): padding index. Default: 0
    Returns:
        (torch.BoolTensor): Mask with shape (B, T_q, T_k). True element stands for requiring making.
    Notes:
        Assume padding_index is 0:
        k.eq(0) -> BoolTensor (B, T_k)
        k.eq(0).unsqueeze(1)  -> (B, 1, T_k)
        k.eq(0).unsqueeze(1).expand(-1, q.size(-1), -1) -> (B, T_q, T_k)
    """


    q = torch.mean(q,2)

    mask = k.eq(padding_index).unsqueeze(1).expand(-1, q.size(-1), -1)
    return mask


class ScaledDotProductAttention(nn.Module):
    """Scaled dot-product attention calculation"""

    def __init__(self, num_heads = 2, dropout_rate=0.0, **kwargs):
        """Initialize ScaledDotProductAttention
        Args:
            dropout_rate (float): attention dropout_rate rate
        """
        super().__init__()
        print("inside scaled dot prod attention")

        self.dropout = nn.Dropout(dropout_rate)
        # self.merged_heads = nn.Linear(num_heads, 1)




    def forward(self, Q, K, V, attn_mask = None):

      scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(K.size(-1)) # scores : [batch_size x n_heads x len_q(=len_k) x len_k(=len_q)]
      scores.masked_fill_(attn_mask, -1e9) # Fills elements of self tensor with value where mask is one.
      attn = nn.Softmax(dim=-1)(scores)
      attn = self.dropout(attn)
      context = torch.matmul(attn, V)

      # attention_weights = self.merged_heads(attn.permute(0,2,3,1))
      return context, attn

class MultiHeadAttention(nn.Module):

    def __init__(self, model_dim=512, num_heads=4, dropout_rate=0.0, attention_type='scaled_dot', query_key_value_weights = [], device = "cuda"):
        super().__init__()
        assert model_dim % num_heads == 0, 'model_dim should be devided by num_heads'
        self.h_size = model_dim
        self.num_heads = num_heads
        self.device = device
        self.head_h_size = model_dim // num_heads

        self.linear_q = nn.Linear(self.h_size, self.h_size)
        self.linear_k = nn.Linear(self.h_size, self.h_size)
        self.linear_v = nn.Linear(self.h_size, self.h_size)
        self.fc0 = nn.Linear(self.h_size, self.h_size)

        ## positional encoding to be added
        # self.positional_encoder = PositionalEncoding(self.h_size).to(self.device)
        ### newly added dropout
        self.attention = ScaledDotProductAttention(dropout_rate = 0.2).to(self.device)
        # self.attention = CosineAttention(dropout_rate = 0.2)
        self.dropout = nn.Dropout(dropout_rate)
        self.lnorm = nn.LayerNorm(model_dim)



    def forward(self, q, k, v, attn_mask=None):
        batch_size = q.size(0)

        tensor1 = []


        # Residual
        residual = q

        # q, k, v = self.add_positional_mask(q, k, v)

        # Linear projection
        q = self.linear_q(q)
        k = self.linear_k(k)
        v = self.linear_v(v)

        # Form multi heads
        q = q.view(batch_size, -1, self.num_heads, self.head_h_size).transpose(1,2)  # (h * B, T_q, D / h)
        k = k.view(batch_size, -1, self.num_heads, self.head_h_size).transpose(1,2)  # (h * B, T_k, D / h)
        v = v.view(batch_size, -1, self.num_heads, self.head_h_size).transpose(1,2)  # (h * B, T_v, D / h)

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1).to(self.device)  # (h * B, T_q, T_k)



        context, attention_per_head = self.attention(q, k, v, attn_mask=attn_mask)
        # context: (h * B, T_q, D_v) attention: (h * B, T_q, T_k)

        # Concatenate heads
        context = context.view(batch_size, -1, self.h_size)  # (B, T_q, D)


        # Dropout
        output = self.dropout(self.fc0(context))  # (B, T_q, D)

        # Residual connection and Layer Normalization
        output = self.lnorm(residual + output)  # (B, T_q, D)

        return output



'''
    MTL Model to classify. Our Architecture which used the RR component and
    Shift component parallely to get the emission scores and then they are
    fed into the CRF to get the appropriate probabilities for each label.
'''
class MTL_Classifier(nn.Module):
    def __init__(self, n_tags, sent_emb_dim, sos_tag_idx, eos_tag_idx, pad_tag_idx, vocab_size = 0, pad_word_idx = 0, pretrained = False, device = 'cuda', use_attention = False):
        super().__init__()

        self.emb_dim = sent_emb_dim
        self.pretrained = pretrained
        self.use_attention = use_attention
        self.device = device
        self.pad_tag_idx = pad_tag_idx
        self.pad_word_idx = pad_word_idx


        ##### attention code ##########
        if self.use_attention:

          attention_type = "scaled_dot"
          self.attention_heads = 8
          self.num_blocks = 2
          self.dropout_rate = 0.2


          multi_headed_attention_weights = []

          for i in range(self.num_blocks):
              self.__setattr__('multihead_attn_{}'.format(i), MultiHeadAttention(model_dim=self.emb_dim,
                                                                                num_heads=self.attention_heads,
                                                                                dropout_rate=self.dropout_rate,
                                                                                attention_type=attention_type,
                                                                                query_key_value_weights = multi_headed_attention_weights,
                                                                                device=self.device))

        ##### attention code ##########


        if not self.pretrained:


          self.rhetorical_encoder = AutoModel.from_pretrained('nlpaueb/legal-bert-small-uncased')
          count = 0
          for name, param in (self.rhetorical_encoder).named_parameters():
              # if
              count = count + 1
              if count <= 84:
                  param.requires_grad = False
              else:
                  param.requires_grad = True



        ## RR Model
        self.emitter = LSTM_Emitter(n_tags, sent_emb_dim, sent_emb_dim, 0.5, self.device).to(self.device)
        self.crf = CRF(n_tags, sos_tag_idx, eos_tag_idx, pad_tag_idx).to(self.device)

        ## Shift or Binary Module
        self.emitter_binary = LSTM_Emitter_Binary(5, 2*sent_emb_dim, sent_emb_dim, 0.5, self.device).to(self.device)
        self.crf_binary = CRF(5, sos_tag_idx, eos_tag_idx, pad_tag_idx).to(self.device)


    def forward(self, x, x_binary, y=[]):
        batch_size = len(x)
        seq_lengths = [len(doc) for doc in x]
        max_seq_len = max(seq_lengths)



        if not self.pretrained:  ## x: list[batch_size, sents_per_doc, words_per_sent]
            # tensor_x = self.encoder(x)
            tensor_x = []
            for doc in x:


                sents = [torch.tensor(s, dtype=torch.long) for s in doc]
                sent_lengths = [len(s) for s in doc]

                ## list[sents_per_doc, words_per_sent] --> tensor[sents_per_doc, max_sent_len]
                sents = nn.utils.rnn.pad_sequence(sents, batch_first=True, padding_value=self.pad_word_idx).to(
                    self.device)

                hidden_reps = self.rhetorical_encoder(sents)
                hidden = hidden_reps[0][:, 0, :self.emb_dim]
                tensor_x.append(hidden)
        else:  ## x: list[batch_size, sents_per_doc, sent_emb_dim]
            tensor_x = [torch.tensor(doc, dtype=torch.float, requires_grad=True) for doc in x]


        tensor_x_binary = [torch.tensor(doc, dtype = torch.float, requires_grad = True) for doc in x_binary]

        tensor_x = nn.utils.rnn.pad_sequence(tensor_x, batch_first = True).to(self.device)

        tensor_x_binary = nn.utils.rnn.pad_sequence(tensor_x_binary, batch_first = True).to(self.device)




        self.emissions_binary, self.hidden_binary = self.emitter_binary(tensor_x_binary, tensor_x)


        self.mask = torch.zeros(batch_size, max_seq_len).to(self.device)
        for i, sl in enumerate(seq_lengths):
            self.mask[i, :sl] = 1


        if self.use_attention:
          ######### added attention on the ouptput ##################


          # if y !=[]:
          y_list = list(zip(*itertools.zip_longest(*y, fillvalue=0)))
          y_tensor = torch.as_tensor(y_list)
          attn_mask = attention_padding_mask(tensor_x, y_tensor, padding_index=0)  # (B, T, T)
          attn_mask = attn_mask.to(self.device)

          for i in range(self.num_blocks):
              tensor_x = self.__getattr__('multihead_attn_{}'.format(i))(tensor_x, tensor_x, tensor_x,
                                                                            attn_mask=attn_mask)  # (B, T, D)


          self.mask = torch.zeros(batch_size, max_seq_len).to(self.device)
          for i, sl in enumerate(seq_lengths):
              self.mask[i, :sl] = 1


          ########## added attention on the output ################




        ## Get hidden states of Shift Module and pass them to the RR Module for emission score calculation for RR Module


        self.emissions = self.emitter(tensor_x, self.hidden_binary)
        # self.emissions = self.emitter(tensor_x, [])

        ## Passing the emission scores to the CRF to get the final sequence of tags
        _, path = self.crf.decode(self.emissions, mask = self.mask)
        _, path_binary = self.crf_binary.decode(self.emissions_binary, mask = self.mask)
        return path, path_binary

    def _loss(self, y):
        ##  list[batch_size, sents_per_doc] --> tensor[batch_size, max_seq_len]
        tensor_y = [torch.tensor(doc, dtype = torch.long) for doc in y]
        tensor_y = nn.utils.rnn.pad_sequence(tensor_y, batch_first = True, padding_value = self.pad_tag_idx).to(self.device)

        nll = self.crf(self.emissions, tensor_y, mask = self.mask)
        return nll

    def _loss_binary(self, y_binary):
        ##  list[batch_size, sents_per_doc] --> tensor[batch_size, max_seq_len]
        tensor_y_binary = [torch.tensor(doc, dtype = torch.long) for doc in y_binary]
        tensor_y_binary = nn.utils.rnn.pad_sequence(tensor_y_binary, batch_first = True, padding_value = self.pad_tag_idx).to(self.device)

        nll_binary = self.crf_binary(self.emissions_binary, tensor_y_binary, mask = self.mask)
        return nll_binary

from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained('nlpaueb/legal-bert-small-uncased')
encoder = AutoModel.from_pretrained('nlpaueb/legal-bert-small-uncased')


def prepare_folds(args):
    with open(args.cat_path) as fp:

        categories = []
        for line in fp:
            _, docs = line.strip().split('\t')
            docs = docs.strip().split(' ')
            categories.append(docs)

    # categories: list[category, docs_per_category]

    categories.sort(key = lambda x: len(x))
    n_docs = len(sum(categories, []))
    print(n_docs)
    assert n_docs == args.dataset_size, "invalid category list"

    docs_per_fold = args.dataset_size // args.num_folds
    folds = [[] for f in range(docs_per_fold)]
    print(folds)

    # folds: list[num_folds, docs_per_fold]

    f = 0
    for cat in categories:
        for doc in cat:
            folds[f].append(doc)
            f = (f + 1) % 5

    # list[num_folds, docs_per_fold] --> list[num_folds * docs_per_fold]
    idx_order = sum(folds, [])
    return idx_order


'''
    This function prepares the numericalized data in the form of lists, to be used for training, test and evaluation.
        x:  list[num_docs, sentences_per_doc, sentence_embedding_dim]
        y:  list[num_docs, sentences_per_doc]
'''
def prepare_data_new(idx_order, args, data_path, tag2idx=None, data_type = 'rhetoric'):
    x, y = [], []



    word2idx = defaultdict(lambda: len(word2idx))
    if tag2idx is None:
        tag2idx = defaultdict(lambda: len(tag2idx))
        tag2idx['<pad>'], tag2idx['<start>'], tag2idx['<end>'] = 0, 1, 2


    # map the special symbols first
    if data_type == 'binary':
      word2idx['<pad>'], word2idx['<unk>'] = 0, 1
    else:
      word2idx['<pad>'], word2idx['<unk>'], word2idx['[CLS]'], word2idx['[SEP]'] = 0, 1, 2, 3


    # iterate over documents
    for doc in idx_order:
        doc_x, doc_y = [], []


        if data_type == 'binary':

          with open(data_path + doc + '.txt') as fp:
            full_curr_doc = fp.readlines()
            for sent_num, sent in enumerate(full_curr_doc):
                    # if sent_num == 0:
                    #   continue
                    sent_x, sent_y = sent.strip().split('\t')
                    if 'tensor(0)' in sent_y:
                      sent_y = '0'
                    elif 'tensor(1)' in sent_y:
                      sent_y = '1'

                    sent_x_curr_and_next = list(map(float, sent_x.strip().split()[:args.shift_emb_dim]))
                    # sent_x_curr_and_prev = list(map(float, full_curr_doc[sent_num - 1].strip().split()[:args.emb_dim]))
                    sent_x = sent_x_curr_and_next
                    # + sent_x_curr_and_prev
                    sent_y = tag2idx[str(sent_y).strip()]

                    if sent_x != []:
                        doc_x.append(sent_x)
                        doc_y.append(sent_y)

        else:
          with open(data_path + doc + '.txt') as fp:

              # iterate over sentences
              for sent in fp:
                  try:
                    sent_x, sent_y = sent.strip().split('\t')
                  except ValueError:
                    continue

                  # cleanse text, map words and tags
                  if not args.pretrained:

                    tokens = tokenizer.tokenize(sent_x)

                    if len(tokens) >= 50:
                        tokens = tokens[0:50]
                    tokens = ['[CLS]'] + tokens + ['[SEP]']
                    sent_x = tokenizer.convert_tokens_to_ids(tokens)


                      # sent_x = sent_x.strip().lower().translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
                      # sent_x = list(map(lambda x: word2idx[x], sent_x.split()))

                  else:
                      sent_x = list(map(float, sent_x.strip().split()[:args.emb_dim]))
                  sent_y = tag2idx[str(sent_y).strip()]

                  if sent_x != []:
                      doc_x.append(sent_x)
                      doc_y.append(sent_y)

        # if data_type == 'rhetoric':
        #   doc_x, doc_y = doc_x[:-1], doc_y[:-1]

        x.append(doc_x)
        y.append(doc_y)


    return x, y,  word2idx, tag2idx

'''To create batches for training'''
def batchify(x, y, x_binary, y_binary, batch_size):
    idx = list(range(len(x)))
    random.shuffle(idx)

    # convert to numpy array for ease of indexing
    x = np.array(x)[idx]
    y = np.array(y)[idx]

    x_binary = np.array(x_binary)[idx]
    y_binary = np.array(y_binary)[idx]

    i = 0
    while i < len(x):
        j = min(i + batch_size, len(x))

        batch_idx = idx[i : j]
        batch_x = x[i : j]
        batch_y = y[i : j]

        batch_x_binary = x_binary[i : j]
        batch_y_binary = y_binary[i : j]

        yield batch_idx, batch_x, batch_y, batch_x_binary, batch_y_binary

        i = j

'''
    Perform a single training step by iterating over the entire training data once. Data is divided into batches.
'''
def train_step(model, opt, x, y, x_binary, y_binary, batch_size):
    ## x: list[num_examples, sents_per_example, features_per_sentence]
    ## y: list[num_examples, sents_per_example]

    model.train()

    total_loss = 0
    total_rhet_loss = 0
    total_binary_loss = 0
    y_pred = [] # predictions
    y_gold = [] # gold standard
    y_pred_binary = []
    y_gold_binary = []
    idx = [] # example index
    mu = 0.3 # hyperparameter
    for i, (batch_idx, batch_x, batch_y, batch_x_binary, batch_y_binary) in enumerate(batchify(x, y, x_binary, y_binary, batch_size)):
        pred, pred_binary = model(batch_x, batch_x_binary, batch_y)
        loss = model._loss(batch_y)
        loss_binary = model._loss_binary(batch_y_binary)

        overall = torch.add(torch.mul(loss, (1-mu)), torch.mul(loss_binary, mu))

        opt.zero_grad()
        # loss.backward()
        overall.backward()
        opt.step()

        total_loss += overall.item()
        total_rhet_loss += loss.item()
        total_binary_loss += loss_binary.item()
        # total_loss += loss.item()

        y_pred.extend(pred)
        y_gold.extend(batch_y)
        y_pred_binary.extend(pred_binary)
        y_gold_binary.extend(batch_y_binary)
        idx.extend(batch_idx)

    assert len(sum(y, [])) == len(sum(y_pred, [])), "Mismatch in predicted"

    return total_loss / (i + 1), idx, y_gold, y_pred, y_gold_binary, y_pred_binary, total_rhet_loss/ (i + 1), total_binary_loss/ (i + 1)

'''
    Perform a single evaluation step by iterating over the entire training data once. Data is divided into batches.
'''
def val_step(model, x, y, x_binary, y_binary, batch_size):
    ## x: list[num_examples, sents_per_example, features_per_sentence]
    ## y: list[num_examples, sents_per_example]
    ## Similarly for Binary data

    model.eval()

    total_loss = 0
    total_rhet_loss = 0
    total_binary_loss = 0
    y_pred = [] # predictions
    y_gold = [] # gold standard
    y_pred_binary = []
    y_gold_binary = []
    idx = [] # example index
    mu = 0.3

    for i, (batch_idx, batch_x, batch_y, batch_x_binary, batch_y_binary) in enumerate(batchify(x, y, x_binary, y_binary, batch_size)):
        pred, pred_binary = model(batch_x, batch_x_binary, batch_y)

        loss = model._loss(batch_y)
        loss_binary = model._loss_binary(batch_y_binary)

        overall = torch.add(torch.mul(loss, (1-mu)), torch.mul(loss_binary, mu))

        total_loss += overall.item()
        total_rhet_loss += loss.item()
        total_binary_loss += loss_binary.item()

        y_pred.extend(pred)
        y_gold.extend(batch_y)
        y_pred_binary.extend(pred_binary)
        y_gold_binary.extend(batch_y_binary)
        idx.extend(batch_idx)

    assert len(sum(y, [])) == len(sum(y_pred, [])), "Mismatch in predicted"

    return total_loss / (i + 1), idx, y_gold, y_pred, y_gold_binary, y_pred_binary, total_rhet_loss/ (i + 1), total_binary_loss/ (i + 1)
'''
    Report all metrics in format using sklearn.metrics.classification_report
'''
def statistics(data_state, tag2idx):
    idx, gold, pred = data_state['idx'], data_state['gold'], data_state['pred']

    rev_tag2idx = {v: k for k, v in tag2idx.items()}
    tags = [rev_tag2idx[i] for i in range(len(tag2idx)) if rev_tag2idx[i] not in ['<start>', '<end>', '<pad>']]

    # flatten out
    gold = sum(gold, [])
    pred = sum(pred, [])


    print(classification_report(gold, pred, target_names = tags, digits = 3))

fold_num = 0

shift_emb_dim_model = 512

class Args:
    pretrained = False
    use_attention = True
    data_path_rr = '/content/drive/MyDrive/IIT_law_ai/semantic_segmentation/dataset/UK-train-set/' ## Input to the pre=trained embedding(should contain 4 sub-folders, IT test and train, CL test and train)
    data_path_binary = '/content/drive/MyDrive/IIT_law_ai/semantic_segmentation/MTL_SHIFT_MODEL/dataset/uk_models/uk_model_legalbert/'
    save_path = '/content/drive/MyDrive/IIT_law_ai/semantic_segmentation/MTL_SHIFT_MODEL/saved_models/best_models/UK/' ## path to save the model
    cat_path = '/content/drive/MyDrive/IIT_law_ai/semantic_segmentation/uk_categories.txt'
    device = 'cuda' ## device to use
    dataset_size = 50
    emb_dim = 512

    num_folds = 5
    batch_size = 4 ## batch size
    print_every = 1 ## print loss after these many number of epochs
    lr = 0.001 ## learning rate
    reg = 0 ## weight decay for Adam Opt
    shift_emb_dim = shift_emb_dim_model ## the pre-trained embedding dimension of the sentences
    epochs = 50 ## Something between 250-300
args = Args()
print(args.data_path_binary)

idx_order = prepare_folds(args)
# x, y, word2idx, tag2idx = prepare_data_new(idx_order, args, data_path = args.data_path_binary, data_type = 'binary')

'''
    Train the model on entire dataset and report loss and macro-F1 after each epoch.
'''
def learn(model, x_rhet, y_rhet, tag2idx_rhet, x_binary, y_binary, tag2idx_binary, args, val_fold = 0, idx_order = []):

    assert idx_order != [], "empty idx order used"

    samples_per_fold = args.dataset_size // args.num_folds

    val_idx = list(range(val_fold * samples_per_fold, val_fold * samples_per_fold + samples_per_fold))
    train_idx = list(range(val_fold * samples_per_fold)) + list(range(val_fold * samples_per_fold + samples_per_fold, args.dataset_size))

    train_x_rhet = [x_rhet[i] for i in train_idx]
    train_y_rhet = [y_rhet[i] for i in train_idx]
    val_x_rhet = [x_rhet[i] for i in val_idx]
    val_y_rhet = [y_rhet[i] for i in val_idx]

    train_x_binary = [x_binary[i] for i in train_idx]
    train_y_binary = [y_binary[i] for i in train_idx]
    val_x_binary = [x_binary[i] for i in val_idx]
    val_y_binary = [y_binary[i] for i in val_idx]


    opt = torch.optim.Adam(model.parameters(), lr = args.lr, weight_decay = args.reg)

    print("{0:>7}  {1:>10}  {2:>6}  {3:>10}  {4:>6}  {5:>10}  {6:>6}  {7:>10}  {8:>6}".format('EPOCH', 'Tr_LOSS', 'Tr_F1', 'Val_LOSS', 'Val_F1', 'Tr_bin_loss', 'Tr_bin_F1', 'Val_bin_loss' ,'Val_bin_F1'))
    print("--------------------------------------------------------------------------------------")

    best_val_f1 = 0

    model_state = {}
    data_state = {}
    start_time = time.time()
    prev_val_f1 = 0
    decrement_ctr = 0

    for epoch in range(1, args.epochs + 1):

        train_loss, train_idx, train_gold, train_pred, train_gold_binary, train_pred_binary, train_rhet_loss, train_binary_loss = train_step(model, opt, train_x_rhet, train_y_rhet, train_x_binary, train_y_binary, args.batch_size)
        val_loss, val_idx, val_gold, val_pred, val_gold_binary, val_pred_binary, val_rhet_loss, val_binary_loss = val_step(model, val_x_rhet, val_y_rhet, val_x_binary, val_y_binary, args.batch_size)

        train_f1 = f1_score(sum(train_gold, []), sum(train_pred, []), average = 'macro')
        val_f1 = f1_score(sum(val_gold, []), sum(val_pred, []), average = 'macro')
        train_f1_binary = f1_score(sum(train_gold_binary, []), sum(train_pred_binary, []), average = 'macro')
        val_f1_binary = f1_score(sum(val_gold_binary, []), sum(val_pred_binary, []), average = 'macro')

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            model_state = {'epoch': epoch, 'arch': model, 'name': model.__class__.__name__, 'state_dict': model.state_dict(), 'best_f1': val_f1, 'optimizer' : opt.state_dict()}
            data_state = {'idx': val_idx, 'loss': val_loss, 'gold': val_gold, 'pred': val_pred}

        if epoch % args.print_every == 0:
          print("{0:7d}  {1:10.3f}  {2:6.3f}  {3:10.3f}  {4:6.3f} {5:10.3f}  {6:6.3f}  {7:10.3f}  {8:6.3f}".format(epoch, train_loss, train_f1, val_loss, val_f1, train_binary_loss, train_f1_binary, val_binary_loss, val_f1_binary))

          if val_f1 < prev_val_f1 and decrement_ctr>3:
            break
          elif val_f1 < prev_val_f1:
            prev_val_f1 = val_f1
            decrement_ctr+=1


    end_time = time.time()

    print("Dumping model and data ...", end = ' ')
    torch.save(model_state, args.save_path + 'model_state' + str(val_fold) + '.tar')

    with open(args.save_path + 'data_state' + str(val_fold) + '.json', 'w') as fp:
        json.dump(data_state, fp)


    print("Done")
    print('Time taken:', int(end_time - start_time), 'secs')

    ## Getting results on Val data
    statistics(data_state, tag2idx)


np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)


print('\nPreparing data ...', end = ' ')

x_rhet, y_rhet, word2idx, tag2idx = prepare_data_new(idx_order, args, data_path = args.data_path_rr)

## Emb dim is 3 times because input for shift module -> concat(shift embedding of current and previous sentence, pre-trained emb of curr sentence(For RR module this is the only input), shift emb of current and next sentence)
x_binary, y_binary, word2idx_binary, tag2idx_binary = prepare_data_new(idx_order, args, data_path = args.data_path_binary, data_type = 'binary')



print('Done')

print('#Tags Overall:', len(tag2idx))

print('#Tags Overall binary:', len(tag2idx_binary))

print('Dump word2idx and tag2idx')
with open(args.save_path + 'word2idx.json', 'w') as fp:
    json.dump(word2idx, fp)
with open(args.save_path + 'tag2idx.json', 'w') as fp:
    json.dump(tag2idx, fp)

with open(args.save_path + 'word2idx_binary.json', 'w') as fp:
    json.dump(word2idx_binary, fp)
with open(args.save_path + 'tag2idx_binary.json', 'w') as fp:
    json.dump(tag2idx_binary, fp)



for fold in range(fold_num,fold_num+1):
  print('\nInitializing model for Overall ...', end = ' ')
  model = MTL_Classifier(len(tag2idx), args.emb_dim, tag2idx['<start>'], tag2idx['<end>'], tag2idx['<pad>'], vocab_size = len(word2idx), pretrained = args.pretrained, device = args.device, use_attention = args.use_attention).to(args.device)
  print('Done')
  print('\nEvaluating on test...')
  print("running fold {}".format(fold))
  learn(model, x_rhet, y_rhet, tag2idx, x_binary, y_binary, tag2idx_binary, args, idx_order = idx_order, val_fold = fold)
