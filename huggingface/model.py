from transformers import (
    BertPreTrainedModel,
    DistilBertPreTrainedModel,
    RobertaModel,
    BertModel,
    AlbertModel,
    # BertLMHeadModel,
    DistilBertModel
    # BertForQuestionAnswering
)
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch


def ce_loss(
        pred, truth, smoothing=False, neighbour_smoothing=False, trg_pad_idx=-1, eps=0.1
):
    truth = truth.contiguous().view(-1)

    one_hot = torch.zeros_like(pred).scatter(1, truth.view(-1, 1), 1)
    one_hot_ = one_hot.clone()

    if smoothing:
        n_class = pred.size(1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)

        if neighbour_smoothing:
            n = 1
            for i in range(1, n):
                one_hot[:, :-i] += ((n - i) * eps) * one_hot_[:, i:]
                one_hot[:, i:] += ((n - i) * eps) * one_hot_[:, :-i]
            one_hot = one_hot / one_hot.sum(1, keepdim=True)

    loss = -one_hot * F.log_softmax(pred, dim=1)

    if trg_pad_idx >= 0:
        loss = loss.sum(dim=1)
        non_pad_mask = truth.ne(trg_pad_idx)
        loss = loss.masked_select(non_pad_mask)

    return loss.sum()


def loss_fn(start_logits, end_logits, start_positions, end_positions):
    config = {
        "smoothing": True,
        "neighbour_smoothing": False,
        "eps": 0.1,
        "use_dist_loss": False,
        "dist_loss_weight": 1,
    }

    bs = start_logits.size(0)

    start_loss = ce_loss(
        start_logits,
        start_positions,
        smoothing=config["smoothing"],
        eps=config["eps"],
        neighbour_smoothing=config["neighbour_smoothing"],
    )

    end_loss = ce_loss(
        end_logits,
        end_positions,
        smoothing=config["smoothing"],
        eps=config["eps"],
        neighbour_smoothing=config["neighbour_smoothing"],
    )

    total_loss = start_loss + end_loss

    return total_loss / bs


class RobertaForQuestionAnswering(BertPreTrainedModel):

    def __init__(self, config):
        super(RobertaForQuestionAnswering, self).__init__(config)
        self.roberta = RobertaModel(config)
        self.conv = nn.Conv1d(config.hidden_size * 3, config.hidden_size, kernel_size=3, padding=(3 - 1) // 2)
        self.fc = nn.Linear(config.hidden_size, 2)
        self.high_dropout = nn.ModuleList([nn.Dropout(p) for p in np.linspace(0.1, 0.5, 5)])

    def forward(self, input_ids, attention_mask, start_positions=None, end_positions=None, use_jaccard_soft=False):
        _, _, hidden_states = self.roberta(input_ids, attention_mask)
        out = torch.cat([hidden_states[-1], hidden_states[-2], hidden_states[-3]], dim=-1).permute(0, 2, 1)
        out = self.conv(out).permute(0, 2, 1)
        out = F.relu(out)
        # logits = self.fc(out)
        all_logits = []
        for dropout in self.high_dropout:
            drop_out = dropout(out)
            logits = self.fc(drop_out)
            all_logits.append(logits)
        logits = torch.stack(all_logits, dim=2).mean(dim=2)

        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)

            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            if use_jaccard_soft:
                start_positions = start_positions.type(start_logits.dtype)
                end_positions = end_positions.type(end_logits.dtype)
                start_log_probs = torch.log_softmax(start_logits, dim=-1)
                end_log_probs = torch.log_softmax(end_logits, dim=-1)
                criterion = nn.KLDivLoss()
                assert start_log_probs.dtype == start_positions.dtype, (start_log_probs.dtype, start_positions.dtype)
                start_loss = criterion(start_log_probs, start_positions)
                end_loss = criterion(end_log_probs, end_positions)
            else:
                criterion = nn.CrossEntropyLoss(ignore_index=ignored_index)
                start_loss = criterion(start_logits, start_positions)
                end_loss = criterion(end_logits, end_positions)

            total_loss = (start_loss + end_loss) / 2

            return total_loss

        else:
            return start_logits, end_logits


class DistilBertForQuestionAnswering(DistilBertPreTrainedModel):

    def __init__(self, config):
        super(DistilBertForQuestionAnswering, self).__init__(config)
        self.distilbert = DistilBertModel(config)
        self.conv = nn.Conv1d(config.hidden_size * 3, config.hidden_size, kernel_size=3, padding=(3 - 1) // 2)
        self.fc = nn.Linear(config.hidden_size, 2)
        self.high_dropout = nn.ModuleList([nn.Dropout(p) for p in np.linspace(0.1, 0.5, 5)])

    def forward(self, input_ids, attention_mask, token_type_ids=None,
                start_positions=None, end_positions=None, use_jaccard_soft=False):
        hidden_states = self.distilbert(input_ids, attention_mask, token_type_ids)[-1]
        out = torch.cat([hidden_states[-1], hidden_states[-2], hidden_states[-3]], dim=-1).permute(0, 2, 1)
        out = self.conv(out).permute(0, 2, 1)
        out = F.relu(out)
        # logits = self.fc(out)
        all_logits = []
        for dropout in self.high_dropout:
            drop_out = dropout(out)
            logits = self.fc(drop_out)
            all_logits.append(logits)
        logits = torch.stack(all_logits, dim=2).mean(dim=2)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)

            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            if use_jaccard_soft:
                start_positions = start_positions.type(start_logits.dtype)
                end_positions = end_positions.type(end_logits.dtype)
                start_log_probs = torch.log_softmax(start_logits, dim=-1)
                end_log_probs = torch.log_softmax(end_logits, dim=-1)
                criterion = nn.KLDivLoss()
                assert start_log_probs.dtype == start_positions.dtype, (start_log_probs.dtype, start_positions.dtype)
                start_loss = criterion(start_log_probs, start_positions)
                end_loss = criterion(end_log_probs, end_positions)
            else:
                criterion = nn.CrossEntropyLoss(ignore_index=ignored_index)
                start_loss = criterion(start_logits, start_positions)
                end_loss = criterion(end_logits, end_positions)

            total_loss = (start_loss + end_loss) / 2

            return total_loss

        else:
            return start_logits, end_logits


class BertForQuestionAnswering(BertPreTrainedModel):

    def __init__(self, config):
        super(BertForQuestionAnswering, self).__init__(config)
        self.bert = BertModel(config)
        self.conv = nn.Conv1d(config.hidden_size * 3, config.hidden_size, kernel_size=3, padding=(3 - 1) // 2)
        self.fc = nn.Linear(config.hidden_size, 2)
        self.high_dropout = nn.ModuleList([nn.Dropout(p) for p in np.linspace(0.1, 0.5, 5)])

    def forward(self, input_ids, attention_mask, token_type_ids=None,
                start_positions=None, end_positions=None, use_jaccard_soft=False):
        hidden_states = self.bert(input_ids, attention_mask, token_type_ids)[-1]
        out = torch.cat([hidden_states[-1], hidden_states[-2], hidden_states[-3]], dim=-1).permute(0, 2, 1)
        out = self.conv(out).permute(0, 2, 1)
        out = F.relu(out)
        all_logits = []
        for dropout in self.high_dropout:
            drop_out = dropout(out)
            logits = self.fc(drop_out)
            all_logits.append(logits)
        logits = torch.stack(all_logits, dim=2).mean(dim=2)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)

            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            if use_jaccard_soft:
                start_positions = start_positions.type(start_logits.dtype)
                end_positions = end_positions.type(end_logits.dtype)
                start_log_probs = torch.log_softmax(start_logits, dim=-1)
                end_log_probs = torch.log_softmax(end_logits, dim=-1)
                criterion = nn.KLDivLoss()
                assert start_log_probs.dtype == start_positions.dtype, (start_log_probs.dtype, start_positions.dtype)
                start_loss = criterion(start_log_probs, start_positions)
                end_loss = criterion(end_log_probs, end_positions)
            else:
                criterion = nn.CrossEntropyLoss(ignore_index=ignored_index)
                start_loss = criterion(start_logits, start_positions)
                end_loss = criterion(end_logits, end_positions)

            total_loss = (start_loss + end_loss) / 2

            return total_loss

        else:
            return start_logits, end_logits


class AlbertForQuestionAnswering(BertPreTrainedModel):

    def __init__(self, config):
        super(AlbertForQuestionAnswering, self).__init__(config)
        self.albert = AlbertModel(config)
        self.fc = nn.Linear(2 * config.hidden_size, 2)

    def forward(self, input_ids, attention_mask, token_type_ids,
                start_positions=None, end_positions=None, use_jaccard_soft=False):
        _, _, hidden_states = self.albert(input_ids, attention_mask, token_type_ids)
        out = torch.cat([hidden_states[-1], hidden_states[-2]], dim=-1)
        logits = self.fc(out)

        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)

            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            if use_jaccard_soft:
                start_positions = start_positions.type(start_logits.dtype)
                end_positions = end_positions.type(end_logits.dtype)
                start_log_probs = torch.log_softmax(start_logits, dim=-1)
                end_log_probs = torch.log_softmax(end_logits, dim=-1)
                criterion = nn.KLDivLoss()
                assert start_log_probs.dtype == start_positions.dtype, (start_log_probs.dtype, start_positions.dtype)
                start_loss = criterion(start_log_probs, start_positions)
                end_loss = criterion(end_log_probs, end_positions)
            else:
                criterion = nn.CrossEntropyLoss(ignore_index=ignored_index)
                start_loss = criterion(start_logits, start_positions)
                end_loss = criterion(end_logits, end_positions)

            total_loss = (start_loss + end_loss) / 2

            return total_loss

        else:
            return start_logits, end_logits


# class BertweetForQuestionAnswering(BertPreTrainedModel):
#
#     def __init__(self, config):
#         super(BertweetForQuestionAnswering, self).__init__(config)
#         self.bertweet = BertLMHeadModel(config)
#         self.fc = nn.Linear(config.hidden_size * 2, 2)
#
#     def forward(self, input_ids, attention_mask, start_positions=None, end_positions=None):
#         _, _, hidden_states = self.bertweet(input_ids, attention_mask)
#         out = torch.cat([hidden_states[-1], hidden_states[-1]], dim=-1)
#         logits = self.fc(out)
#
#         start_logits, end_logits = logits.split(1, dim=-1)
#         start_logits = start_logits.squeeze(-1)
#         end_logits = end_logits.squeeze(-1)
#
#         if start_positions is not None and end_positions is not None:
#             # If we are on multi-GPU, split add a dimension
#             if len(start_positions.size()) > 1:
#                 start_positions = start_positions.squeeze(-1)
#             if len(end_positions.size()) > 1:
#                 end_positions = end_positions.squeeze(-1)
#             # sometimes the start/end positions are outside our model inputs, we ignore these terms
#             ignored_index = start_logits.size(1)
#             start_positions.clamp_(0, ignored_index)
#             end_positions.clamp_(0, ignored_index)
#
#             loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)
#             start_loss = loss_fct(start_logits, start_positions)
#             end_loss = loss_fct(end_logits, end_positions)
#             total_loss = (start_loss + end_loss) / 2
#
#             return total_loss
#
#         else:
#             return start_logits, end_logits


class QuestionAnswering:

    def __call__(self, model_type):
        if model_type == 'roberta':
            return RobertaForQuestionAnswering
        elif model_type == 'bert':
            return BertForQuestionAnswering
        elif model_type == 'distilbert':
            return DistilBertForQuestionAnswering
        elif model_type == 'albert':
            return AlbertForQuestionAnswering


######################### 2nd level model #########################
class CharRNN(nn.Module):

    def __init__(self, char_vocab_size, char_embed_dim, n_models, lstm_hidden_size, sentiment_dim, encode_size):
        super(CharRNN, self).__init__()
        self.char_embedding = nn.Embedding(num_embeddings=char_vocab_size, embedding_dim=char_embed_dim)
        self.probs_rnn = nn.LSTM(input_size=n_models * 2, hidden_size=lstm_hidden_size,
                                 num_layers=1, batch_first=True, bidirectional=True)
        self.sentiment_embedding = nn.Embedding(num_embeddings=3, embedding_dim=sentiment_dim)
        self.encoder = nn.LSTM(input_size=char_embed_dim + sentiment_dim + lstm_hidden_size * 2,
                               hidden_size=encode_size, num_layers=2, batch_first=True, bidirectional=True)
        # self.fc = nn.Linear(2 * encode_size, 2)
        self.fc = nn.Sequential(
            nn.Linear(2 * encode_size, encode_size),
            nn.ReLU(),  # 比 leaky relu 好
            nn.Linear(encode_size, 2)

        )
        self.dropouts = nn.ModuleList([nn.Dropout(p) for p in np.linspace(0.1, 0.5, 5)])

    def forward(self, start_probs=None, end_probs=None, char_ids=None,
                sentiment_ids=None, start_positions=None, end_positions=None):
        bs, seq_len = char_ids.size()
        probs = torch.stack([start_probs, end_probs], dim=-1)
        probs_emb, (_, _) = self.probs_rnn(probs)
        char_emb = self.char_embedding(char_ids)
        sentiment_emb = self.sentiment_embedding(sentiment_ids).unsqueeze(1).expand(-1, seq_len, -1)
        emb = torch.cat([probs_emb, char_emb, sentiment_emb], dim=-1)
        out, (_, _) = self.encoder(emb)
        # logits = self.fc(out)

        all_logits = []
        for dropout in self.dropouts:
            drop_out = dropout(out)
            logits = self.fc(drop_out)
            all_logits.append(logits)

        logits = torch.stack(all_logits, dim=2).mean(dim=2)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

            return total_loss

        else:
            return start_logits, end_logits


class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 dilation=1, padding='same', use_bn=True):
        super(ConvBlock, self).__init__()
        if padding == 'same':
            padding = kernel_size // 2 * dilation

        if use_bn:
            self.conv = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size,
                          padding, stride, dilation),
                nn.BatchNorm1d(out_channels),
                nn.ReLU()
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size,
                          padding, stride, dilation),
                nn.ReLU()
            )

    def forward(self, x):
        return self.conv(x)


class CharCNN(nn.Module):

    def __init__(self, char_vocab_size, char_embed_dim, n_models, cnn_dim,
                 sentiment_dim, encode_size, kernel_size):
        super(CharCNN, self).__init__()
        self.use_msd = True
        self.char_embedding = nn.Embedding(num_embeddings=char_vocab_size, embedding_dim=char_embed_dim)
        self.sentiment_embedding = nn.Embedding(num_embeddings=3, embedding_dim=sentiment_dim)
        self.probs_cnn = ConvBlock(n_models * 2, cnn_dim, kernel_size, use_bn=True)
        self.cnn = nn.Sequential(
            ConvBlock(char_embed_dim + cnn_dim + sentiment_dim, encode_size, kernel_size, use_bn=True),
            ConvBlock(encode_size, encode_size * 2, kernel_size, use_bn=True),
            ConvBlock(encode_size * 2, encode_size * 4, kernel_size, use_bn=True),
            ConvBlock(encode_size * 4, encode_size * 8, kernel_size, use_bn=True)
        )
        self.logits = nn.Sequential(
            nn.Linear(encode_size * 8, encode_size),
            nn.ReLU(),  # 比 leaky relu 好
            nn.Linear(encode_size, 2)
        )

        # self.high_dropout = nn.Dropout(0.5)
        self.high_dropout = nn.ModuleList([nn.Dropout(p) for p in np.linspace(0.1, 0.5, 5)])

    def forward(self, start_probs=None, end_probs=None, char_ids=None,
                sentiment_ids=None, start_positions=None, end_positions=None):
        seq_len = char_ids.size(1)
        probs = torch.cat([start_probs, end_probs], dim=1)
        probs_emb = self.probs_cnn(probs).permute(0, 2, 1)
        char_emb = self.char_embedding(char_ids)
        sentiment_emb = self.sentiment_embedding(sentiment_ids).unsqueeze(1).expand(-1, seq_len, -1)
        emb = torch.cat([probs_emb, char_emb, sentiment_emb], dim=-1).permute(0, 2, 1)
        features = self.cnn(emb).permute(0, 2, 1)
        # logits = self.logits(features)

        if self.use_msd and self.training:  # 去掉 self.training 没啥影响
            logits = torch.mean(
                torch.stack(
                    [self.logits(self.high_dropout[i](features)) for i in range(5)],
                    dim=0,
                ),
                dim=0,
            )
        else:
            logits = self.logits(features)

        start_logits, end_logits = logits[:, :, 0], logits[:, :, 1]

        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            # total_loss = loss_fn(start_logits, end_logits, start_positions, end_positions)

            return total_loss

        else:
            return start_logits, end_logits


class TweetCharModel(nn.Module):
    def __init__(self, char_vocab_size, char_embed_dim, n_models, lstm_hidden_size, sentiment_dim, encode_size):
        super().__init__()
        self.use_msd = True
        self.char_embeddings = nn.Embedding(num_embeddings=char_vocab_size, embedding_dim=char_embed_dim)
        self.sentiment_embeddings = nn.Embedding(num_embeddings=3, embedding_dim=sentiment_dim)
        self.proba_lstm = nn.LSTM(n_models * 2, lstm_hidden_size, batch_first=True, bidirectional=True)
        self.lstm = nn.LSTM(char_embed_dim + lstm_hidden_size * 2 + sentiment_dim,
                            encode_size, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(encode_size * 2, encode_size, batch_first=True, bidirectional=True)

        self.logits = nn.Sequential(
            nn.Linear(encode_size * 4, encode_size),
            nn.ReLU(),
            nn.Linear(encode_size, 2),
        )

        self.high_dropout = nn.Dropout(p=0.5)

    def forward(self, start_probs=None, end_probs=None, char_ids=None,
                sentiment_ids=None, start_positions=None, end_positions=None):
        bs, T = char_ids.size()
        probas = torch.stack([start_probs, end_probs], -1)
        probas_fts, _ = self.proba_lstm(probas)
        char_fts = self.char_embeddings(char_ids)
        sentiment_fts = self.sentiment_embeddings(sentiment_ids).view(bs, 1, -1)
        sentiment_fts = sentiment_fts.repeat((1, T, 1))

        features = torch.cat([char_fts, sentiment_fts, probas_fts], -1)
        features, _ = self.lstm(features)
        features2, _ = self.lstm2(features)

        features = torch.cat([features, features2], -1)

        if self.use_msd and self.training:
            logits = torch.mean(
                torch.stack(
                    [self.logits(self.high_dropout(features)) for _ in range(5)],
                    dim=0,
                ),
                dim=0,
            )
        else:
            logits = self.logits(features)

        start_logits, end_logits = logits[:, :, 0], logits[:, :, 1]

        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            # total_loss = loss_fn(start_logits, end_logits, start_positions, end_positions)

            return total_loss

        else:
            return start_logits, end_logits


class CharCNNCross(nn.Module):
    """无效，jaccard：71.74, 8 epochs"""

    def __init__(self, char_vocab_size, char_embed_dim, n_models, cnn_dim,
                 sentiment_dim, encode_size, kernel_size):
        super(CharCNNCross, self).__init__()
        self.use_msd = True
        self.char_embedding = nn.Embedding(num_embeddings=char_vocab_size, embedding_dim=char_embed_dim)
        self.sentiment_embedding = nn.Embedding(num_embeddings=3, embedding_dim=sentiment_dim)
        self.probs_cnn = ConvBlock(n_models * 2, cnn_dim, kernel_size, use_bn=True)
        self.cnn = nn.Sequential(
            ConvBlock(char_embed_dim + cnn_dim + sentiment_dim, encode_size, kernel_size, use_bn=True),
            ConvBlock(encode_size, encode_size * 2, kernel_size, use_bn=True),
            ConvBlock(encode_size * 2, encode_size * 4, kernel_size, use_bn=True),
            ConvBlock(encode_size * 4, encode_size * 8, kernel_size, use_bn=True)
        )
        self.start_logits = nn.Sequential(
            nn.Linear(encode_size * 8, encode_size),
            nn.ReLU(),
            nn.Linear(encode_size, 1)
        )
        self.end_logits = nn.Sequential(
            nn.Linear(encode_size * 16, 2 * encode_size),
            nn.ReLU(),
            nn.Linear(2 * encode_size, 1)
        )

        # self.high_dropout = nn.Dropout(0.5)
        self.high_dropout = nn.ModuleList([nn.Dropout(p) for p in np.linspace(0.1, 0.5, 5)])

    def forward(self, start_probs=None, end_probs=None, char_ids=None, sentiment_ids=None,
                start_positions=None, end_positions=None, beam_size=None):
        seq_len = char_ids.size(1)
        probs = torch.stack([start_probs, end_probs], dim=-1).permute(0, 2, 1)
        probs_emb = self.probs_cnn(probs).permute(0, 2, 1)
        char_emb = self.char_embedding(char_ids)
        sentiment_emb = self.sentiment_embedding(sentiment_ids).unsqueeze(1).expand(-1, seq_len, -1)
        emb = torch.cat([probs_emb, char_emb, sentiment_emb], dim=-1).permute(0, 2, 1)
        start_features = self.cnn(emb).permute(0, 2, 1)  # bs, seq_len, features
        feature_size = start_features.size(2)

        if start_positions is not None and end_positions is not None:
            expanded_start_positions = start_positions[:, None, None].expand(-1, -1, feature_size)  # bs, 1, f
            start_end_features = start_features.gather(dim=1, index=expanded_start_positions).expand(-1, seq_len, -1)
            end_features = torch.cat([start_features, start_end_features], dim=-1)
            if self.use_msd:
                start_logits = torch.mean(
                    torch.stack(
                        [self.start_logits(self.high_dropout[i](start_features)) for i in range(5)],
                        dim=0,
                    ),
                    dim=0,
                ).squeeze()
                end_logits = torch.mean(
                    torch.stack(
                        [self.end_logits(self.high_dropout[i](end_features)) for i in range(5)],
                        dim=0,
                    ),
                    dim=0,
                ).squeeze()
            else:
                start_logits = self.start_logits(start_features).squeeze()
                end_logits = self.end_logits(end_features).squeeze()

            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            # total_loss = loss_fn(start_logits, end_logits, start_positions, end_positions)

            return total_loss

        else:
            start_end_idx_map = {}
            total_logits = []
            start_idxs = []
            end_idxs = []
            if self.use_msd:
                start_logits = torch.mean(
                    torch.stack(
                        [self.start_logits(self.high_dropout[i](start_features)) for i in range(5)],
                        dim=0,
                    ),
                    dim=0,
                ).squeeze()
            else:
                start_logits = self.start_logits(start_features).squeeze()

            top_start_logits, top_indices = torch.topk(start_logits, k=beam_size, dim=-1)  # bs, top_k, 要循环
            expand_top_indices = top_indices[:, :, None].expand(-1, -1, feature_size)
            topk_start_end_features = start_features.gather(dim=1, index=expand_top_indices)  # bs, topk, features
            for i in range(beam_size):
                start_end_features = topk_start_end_features[:, [i], :].expand(-1, seq_len, -1)
                current_start_logit = top_start_logits[:, i]
                end_features = torch.cat([start_features, start_end_features], dim=-1)
                if self.use_msd:
                    end_logits = torch.mean(
                        torch.stack(
                            [self.end_logits(self.high_dropout[i](end_features)) for i in range(5)],
                            dim=0,
                        ),
                        dim=0,
                    ).squeeze()
                else:
                    end_logits = self.end_logits(end_features).squeeze()
                max_end_logit, end_idx = end_logits.max(dim=-1)  # bs
                start_end_idx_map.update({i: end_idx})
                total_logit = current_start_logit + max_end_logit
                total_logits.append(total_logit)

            total_logits = torch.stack(total_logits, dim=-1)
            max_start_idxs = total_logits.argmax(dim=-1).tolist()  # bs
            for i, idx in enumerate(max_start_idxs):
                start_idxs.append(top_indices[i, idx].item())
                end_idxs.append(start_end_idx_map[idx][i].item())
            return start_idxs, end_idxs
