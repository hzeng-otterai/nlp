from typing import Dict, Optional

from overrides import overrides

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.data import Vocabulary
from allennlp.modules import FeedForward, Seq2VecEncoder, TextFieldEmbedder
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn import util
from allennlp.training.metrics import CategoricalAccuracy

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .matching_layer import MatchingLayer
eps = 1e-8


class LSTM_V(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, batch_first=True, dropout=0,
                 bidirectional=False, only_use_last_hidden_state=False):
        """
        LSTM which can hold variable length sequence, use like TensorFlow's RNN(input, length...).

        :param input_size:The number of expected features in the input x
        :param hidden_size:The number of features in the hidden state h
        :param num_layers:Number of recurrent layers.
        :param bias:If False, then the layer does not use bias weights b_ih and b_hh. Default: True
        :param batch_first:If True, then the input and output tensors are provided as (batch, seq, feature)
        :param dropout:If non-zero, introduces a dropout layer on the outputs of each RNN layer except the last layer
        :param bidirectional:If True, becomes a bidirectional RNN. Default: False
        """
        super(LSTM_V, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.only_use_last_hidden_state = only_use_last_hidden_state
        self.LSTM = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=bias,
            batch_first=batch_first,
            dropout=dropout,
            bidirectional=bidirectional
        )

    def forward(self, x, x_len):
        """
        sequence -> sort -> pad and pack ->process using RNN -> unpack ->unsort

        :param x: Variable
        :param x_len: numpy list
        :return:
        """
        """sort"""
        x_sort_idx = np.argsort(-x_len)
        x_unsort_idx = np.argsort(x_sort_idx)
        x_len = x_len[x_sort_idx]
        x = x[x_sort_idx]

        """pack"""
        x_emb_p = torch.nn.utils.rnn.pack_padded_sequence(x, x_len, batch_first=self.batch_first)

        """process using RNN"""
        out_pack, (ht, ct) = self.LSTM(x_emb_p, None)

        """unsort: h"""
        # (num_layers * num_directions, batch, hidden_size) -> (batch, ...)
        ht = torch.transpose(ht, 0, 1)[x_unsort_idx]
        ht = torch.transpose(ht, 0, 1)

        if self.only_use_last_hidden_state:
            return None, (ht, None)
        else:
            """unpack: out"""
            out = torch.nn.utils.rnn.pad_packed_sequence(out_pack, batch_first=self.batch_first)  # (sequence, lengths)
            out = out[0]  #
            """unsort: out c"""
            out = out[x_unsort_idx]
            # (num_layers * num_directions, batch, hidden_size) -> (batch, ...)
            ct = torch.transpose(ct, 0, 1)[x_unsort_idx]
            ct = torch.transpose(ct, 0, 1)

            return out, (ht, ct)

@Model.register("BiMPM")
class BiMPM(nn.Module):
    def __init__(self, 
        class_size, word_vocab_size, char_vocab_size,
        pretrained_word_embedding=None, pretrained_char_embedding=None,
        num_perspective=20, word_dim=300, dropout=0.1, fix_word_vec=True,
        use_char_emb=True, char_dim=20, char_lstm_dim=50, fix_char_vec=False,
        context_lstm_dim=100, context_layer_num=2, 
        aggregation_lstm_dim=100, aggregation_layer_num=2, 
        wo_full_match=False, wo_maxpool_match=False, 
        wo_attentive_match=False, wo_max_attentive_match=False
    ):
        super(BiMPM, self).__init__()

        self.class_size = class_size
        self.use_char_emb = use_char_emb
        self.context_lstm_dim = context_lstm_dim
        self.context_layer_num = context_layer_num
        self.aggregation_lstm_dim = aggregation_lstm_dim
        self.aggregation_layer_num = aggregation_layer_num
        self.char_lstm_dim = char_lstm_dim
        self.dropout = dropout
        self.word_rep_dim = word_dim + int(use_char_emb) * 2 * char_lstm_dim
        self.num_perspective = num_perspective
        self.num_matching = 8 - 2 * (int(wo_full_match) + int(wo_maxpool_match) + int(wo_attentive_match) + int(wo_max_attentive_match))
        assert self.num_matching > 0

        # ----- Word Representation Layer -----
        assert pretrained_word_embedding is None or len(pretrained_word_embedding) == word_vocab_size
        self.word_emb = nn.Embedding(word_vocab_size, word_dim, padding_idx=1)
        if fix_word_vec:
            self.word_emb.weight.requires_grad = False

        if use_char_emb:
            assert pretrained_char_embedding is None or len(pretrained_char_embedding) == char_vocab_size
            self.char_emb = nn.Embedding(char_vocab_size, char_dim, padding_idx=1)
            if fix_char_vec:
                self.char_emb.weight.requires_grad = False

            self.char_LSTM = LSTM_V(
                input_size=char_dim,
                hidden_size=char_lstm_dim,
                num_layers=1,
                dropout=self.dropout,
                bidirectional=True,
                batch_first=True)

        # ----- Context Representation Layer -----
        self.context_LSTM = LSTM_V(
            input_size=self.word_rep_dim,
            hidden_size=context_lstm_dim,
            num_layers=self.context_layer_num,
            dropout=self.dropout,
            bidirectional=True,
            batch_first=True
        )

        # ----- Matching Layer -----
        self.matching_layer = MatchingLayer(
            hidden_dim=self.context_lstm_dim, 
            num_perspective=self.num_perspective, 
            dropout=self.dropout,
            wo_full_match=wo_full_match,
            wo_maxpool_match=wo_maxpool_match,
            wo_attentive_match=wo_attentive_match,
            wo_max_attentive_match=wo_max_attentive_match,
        )

        # ----- Aggregation Layer -----
        self.aggregation_LSTM = LSTM_V(
            input_size=num_perspective * self.num_matching,
            hidden_size=aggregation_lstm_dim,
            num_layers=self.aggregation_layer_num,
            dropout=self.dropout,
            bidirectional=True,
            batch_first=True
        )

        # ----- Prediction Layer -----
        self.pred_fc1 = nn.Linear(
            self.aggregation_lstm_dim * 4 * self.aggregation_layer_num, 
            self.aggregation_lstm_dim * 2 * self.aggregation_layer_num
        )
        self.pred_fc2 = nn.Linear(
            self.aggregation_lstm_dim * 2 * self.aggregation_layer_num, 
            self.class_size
        )

        self.init_parameters(pretrained_word_embedding, pretrained_char_embedding)
        
    def init_lstm(self, lstm_layer):
        for name, param in lstm_layer.named_parameters():
            if "weight_ih" in name:
                nn.init.kaiming_normal_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.constant_(param, val=0)

    def init_parameters(self, pretrained_word_embedding, pretrained_char_embedding):
        # ----- Word Representation Layer -----

        # <unk> vectors is randomly initialized
        if pretrained_word_embedding is not None:
            self.word_emb.weight.data.copy_(pretrained_word_embedding)
            nn.init.uniform_(self.word_emb.weight.data[0], -0.005, 0.005)
            for data in self.word_emb.weight.data[2:]:
                if (data.abs() < eps).all():
                    nn.init.uniform_(data, -0.005, 0.005)
        else:
            nn.init.uniform_(self.word_emb.weight, -0.005, 0.005)
            
        self.word_emb.weight.data[1].fill_(0)

        if self.use_char_emb:
            # <unk> vectors is randomly initialized
            if pretrained_char_embedding is not None:
                self.char_emb.weight.data.copy_(pretrained_char_embedding)
                nn.init.uniform_(self.char_emb.weight.data[0], -0.005, 0.005)
                for data in self.char_emb.weight.data[2:]:
                    if (data.abs() < eps).all():
                        nn.init.uniform_(data, -0.005, 0.005)
            else:
                nn.init.uniform_(self.char_emb.weight, -0.005, 0.005)
                
            self.char_emb.weight.data[1].fill_(0)
            
            self.init_lstm(self.char_LSTM)

        # ----- Context Representation Layer -----
        self.init_lstm(self.context_LSTM)

        # ----- Aggregation Layer -----
        self.init_lstm(self.aggregation_LSTM)

        # ----- Prediction Layer ----
        nn.init.uniform_(self.pred_fc1.weight, -0.005, 0.005)
        nn.init.constant_(self.pred_fc1.bias, val=0)

        nn.init.uniform_(self.pred_fc2.weight, -0.005, 0.005)
        nn.init.constant_(self.pred_fc2.bias, val=0)
        
    def char_repr(self, char_data, char_data_len):
        # char_data: (batch, seq_len, max_word_len)
        # char_data_len: (batch, seq_len)
        assert char_data.size(0) == char_data_len.size(0)
        assert char_data.size(1) == char_data_len.size(1)
        seq_len, word_len = char_data.size(1), char_data.size(2)

        # (batch, seq_len, max_word_len) -> (batch * seq_len, max_word_len)
        char_data = char_data.view(-1, word_len)
        
        # (batch * seq_len, max_word_len) -> (batch * seq_len, max_word_len, char_dim)
        char_data = self.char_emb(char_data)

        # (batch, seq_len) -> (batch * seq_len)
        char_data_len = char_data_len.view(-1)

        # (batch * seq_len, max_word_len, char_dim)-> (2, batch * seq_len, char_lstm_dim)
        _, (char_data, _) = self.char_LSTM(char_data, char_data_len)
        
        # (2, batch * seq_len, char_lstm_dim) -> (batch * seq_len, 2, char_lstm_dim)
        char_data = char_data.permute(1, 0, 2).contiguous()

        # (batch * seq_len, 2, char_lstm_dim) -> (batch, seq_len, 2 * char_lstm_dim)
        char_data = char_data.view(-1, seq_len, 2 * self.char_lstm_dim)
        
        return char_data
        
    def context_repr(self, word_data, word_data_len, char_data, char_data_len):
        # (batch, seq_len) -> (batch, seq_len, word_dim)
        context_data = self.word_emb(word_data)

        if self.use_char_emb:
            char_data = self.char_repr(char_data, char_data_len)
            
            # (batch, seq_len, word_dim + char_lstm_dim)
            context_data = torch.cat([context_data, char_data], dim=-1)

        context_data = F.dropout(context_data, p=self.dropout, training=self.training)

        # ----- Context Representation Layer -----
        # (batch, seq_len, context_lstm_dim * 2)
        context_data, _ = self.context_LSTM(context_data, word_data_len)

        context_data = F.dropout(context_data, p=self.dropout, training=self.training)
        
        return context_data

    def forward(self, **kwargs):

        # ----- Context Representation Layer -----

        # input for words: (batch, seq_len)
        # input for chars: (batch, seq_len, word_len)
        p, p_len, h, h_len = kwargs['p'], kwargs['p_len'], kwargs['h'], kwargs['h_len']
        assert p.size(0) == h.size(0)
        batch_size, seq_len_p, seq_len_h = p.size(0), p.size(1), h.size(1)

        if self.use_char_emb:
            char_p, char_p_len, char_h, char_h_len = kwargs['char_p'], kwargs['char_p_len'], kwargs['char_h'], kwargs['char_h_len']
            assert p.size(1) == char_p.size(1) and h.size(1) == char_h.size(1)
        else:
            char_p, char_p_len, char_h, char_h_len = None, None, None, None
        
        # (batch, seq_len) -> (batch, seq_len, context_lstm_dim * 2)
        con_p = self.context_repr(p, p_len, char_p, char_p_len)
        con_h = self.context_repr(h, h_len, char_h, char_h_len)
        assert con_p.size() == (batch_size, seq_len_p, self.context_lstm_dim * 2)
        assert con_h.size() == (batch_size, seq_len_h, self.context_lstm_dim * 2)

        # (batch, seq_len, context_lstm_dim * 2) -> (batch, seq_len, num_perspective * num_matching)
        mv_p, mv_h = self.matching_layer(con_p, con_h)
        assert mv_p.size() == (batch_size, seq_len_p, self.num_perspective * self.num_matching)
        assert mv_h.size() == (batch_size, seq_len_h, self.num_perspective * self.num_matching)

        # ----- Aggregation Layer -----
        # (batch, seq_len, num_perspective * num_matching) -> 
        # (2 * aggregation_layer_num, batch, aggregation_lstm_dim)
        _, (agg_p_last, _) = self.aggregation_LSTM(mv_p, p_len)
        _, (agg_h_last, _) = self.aggregation_LSTM(mv_h, h_len)
        assert agg_p_last.size() == agg_h_last.size() == (2 * self.aggregation_layer_num, batch_size, self.aggregation_lstm_dim)

        # 2 * (2 * aggregation_layer_num, batch, aggregation_lstm_dim) -> 
        # 2 * (batch, 2 * aggregation_layer_num, aggregation_lstm_dim) -> 
        # 2 * (batch, aggregation_lstm_dim * 2 * aggregation_layer_num) -> 
        # (batch, 2 * aggregation_lstm_dim * 2 * aggregation_layer_num)
        x = torch.cat(
            [agg_p_last.permute(1, 0, 2).contiguous().view(-1, self.aggregation_lstm_dim * 2 * self.aggregation_layer_num),
             agg_h_last.permute(1, 0, 2).contiguous().view(-1, self.aggregation_lstm_dim * 2 * self.aggregation_layer_num)], dim=1)
        x = F.dropout(x, p=self.dropout, training=self.training)
        assert x.size() == (batch_size, 2 * self.aggregation_lstm_dim * 2 * self.aggregation_layer_num)

        # ----- Prediction Layer -----
        x = F.tanh(self.pred_fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        assert x.size() == (batch_size, self.aggregation_lstm_dim * 2 * self.aggregation_layer_num)
        
        x = self.pred_fc2(x)
        assert x.size() == (batch_size, self.class_size)

        return x
