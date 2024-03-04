from model.atten_submodules import *
from model.submodels import *
import itertools


'''
    Top-level module which uses a Hierarchical-LSTM-CRF to classify.
    If pretrained = False, each example is represented as a sequence of sentences, which themselves are sequences of word tokens. Individual sentences are passed to LSTM_Sentence_Encoder to generate sentence embeddings. 
    If pretrained = True, each example is represented as a sequence of fixed-length pre-trained sentence embeddings.
    Sentence embeddings are then passed to LSTM_Emitter to generate emission scores, and finally CRF is used to obtain optimal tag sequence. 
    Emission scores are fed to the CRF to generate optimal tag sequence.
'''

class Attn_BiLSTM_CRF(nn.Module):
    def __init__(self, n_tags, sent_emb_dim, sos_tag_idx, eos_tag_idx, pad_tag_idx, vocab_size=0, word_emb_dim=0,
                     pad_word_idx=0, pretrained=False, device='cuda', num_blocks = 4,
                     dropout_rate = 0.2,
                     attention_type = "scaled_dot",
                     attention_heads = 4,
                     feed_forward_hidden_dim = 400):

        super().__init__()

        self.emb_dim = sent_emb_dim
        self.n_tags = n_tags
        self.num_heads = attention_heads
        self.pretrained = pretrained
        self.device = device
        self.pad_tag_idx = pad_tag_idx
        self.pad_word_idx = pad_word_idx
        self.num_blocks = num_blocks


        # self.word_pad_idx = pad_word_idx
        # self.ent_pad_idx = arg.entity_pad[1]
        # self.ent_bos_idx = arg.entity_bos[1]
        # self.ent_eos_idx = arg.entity_eos[1]

        # assert sent_emb_dim == model_dim // 2, 'the output shape of BiGRU should be same as model shape'




        # self.embed = nn.Embedding(arg.num_vocabs, arg.embed_dim)
        # self.emitter = LSTM_Emitter(n_tags, sent_emb_dim, sent_emb_dim).to(self.device)


        self.input_fc = nn.Linear(sent_emb_dim, sent_emb_dim)

        self.gru = nn.LSTM(sent_emb_dim, sent_emb_dim // 2 , batch_first=True, bidirectional=True).to(self.device)

        # for i in range(self.num_blocks):
        #     self.__setattr__('multihead_attn_{}'.format(i), MultiHeadAttention(model_dim=sent_emb_dim,
        #                                                                        num_heads=attention_heads,
        #                                                                        dropout_rate=dropout_rate,
        #                                                                        attention_type=attention_type,
        #                                                                        device = self.device))

        #     self.__setattr__('feedforward_{}'.format(i), FeedForward(model_dim=sent_emb_dim,
        #                                                              hidden_dim=feed_forward_hidden_dim,
        #                                                              dropout_rate=dropout_rate))
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(sent_emb_dim, n_tags)

        self.crf = CRF(n_tags, sos_tag_idx, eos_tag_idx, pad_tag_idx).to(self.device)

    def forward(self, x, y=[]):
        
        """
        Forward logic of model
        Args:
            x (torch.LongTensor): contexts of shape (B, T)
            y (torch.LongTensor): entities of shape (B, T)
        Returns:
            (tuple): tuple containing:
                (torch.Tensor): viterbi score for each sequence in current batch (B,).
                (list[list[int]]): best sequences of entities of this batch, representing in indexes (B, *)
        """
        batch_size = len(x)
        seq_lengths = [len(doc) for doc in x]
        max_seq_len = max(seq_lengths)
        

        tensor_x = [torch.tensor(doc, dtype=torch.float, requires_grad=True) for doc in x]
        tensor_x = nn.utils.rnn.pad_sequence(tensor_x, batch_first=True).to(self.device)


        y_list = list(zip(*itertools.zip_longest(*y, fillvalue=0)))
        y_tensor = torch.as_tensor(y_list)


        attn_mask = attention_padding_mask(tensor_x, y_tensor, padding_index=0)  # (B, T, T)        
        attn_mask = attn_mask.to(self.device)

        ##### ADD POSITION EMBEDDINGS BEFORE SENDING FOR MULTI HEADED ATTENTION 
        # tensor_x = (tensor_x + self.position(tensor_x).to(self.device)).to(self.device)
        # self.emissions = self.emitter(tensor_x, y_tensor, 0)
        attention_per_head  = torch.zeros(self.num_blocks, batch_size, self.num_heads, max_seq_len, max_seq_len)
        # attention_combined  = torch.zeros(self.num_blocks, batch_size, max_seq_len, max_seq_len)

        for i in range(self.num_blocks):
            tensor_x, attention_per_head[i] = self.__getattr__('multihead_attn_{}'.format(i))(tensor_x, tensor_x, tensor_x, attn_mask=attn_mask)  # (B, T, D)
            tensor_x = self.__getattr__('feedforward_{}'.format(i))(tensor_x)  # (B, T, D)
        tensor_x = self.input_fc(tensor_x)
        tensor_x, _ = self.gru(tensor_x)  # x (B, T, 2 * D/2)
        tensor_x = self.dropout(tensor_x)
        tensor_x = tensor_x.to(self.device)
        # print("input to LSTM CRF is on comp graph : "+str(tensor_x.requires_grad))
        self.emissions = self.fc(tensor_x)  # x is now emission matrix (B, T, num_entities)
        # crf_mask = (y_tensor != 0).bool()  # (B, T)

        self.mask = torch.zeros(batch_size, max_seq_len).to(self.device)
        for i, sl in enumerate(seq_lengths):
            self.mask[i, :sl] = 1


        _, path = self.crf.decode(self.emissions, mask=self.mask)



        return path, attention_per_head



    def _loss(self, y):
        ##  list[batch_size, sents_per_doc] --> tensor[batch_size, max_seq_len]
        tensor_y = [torch.tensor(doc, dtype=torch.long) for doc in y]
        tensor_y = nn.utils.rnn.pad_sequence(tensor_y, batch_first=True, padding_value=self.pad_tag_idx).to(self.device)
        self.emissions = self.emissions.to(self.device)
        nll = self.crf(self.emissions, tensor_y, mask=self.mask)


        return nll

