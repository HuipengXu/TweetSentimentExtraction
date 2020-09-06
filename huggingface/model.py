from transformers import (
    BertPreTrainedModel,
    RobertaModel,
    BertModel,
    AlbertModel,
    BertLMHeadModel,
    # BertForQuestionAnswering
)
import torch.nn as nn
import torch


class RobertaForQuestionAnswering(BertPreTrainedModel):

    def __init__(self, config):
        super(RobertaForQuestionAnswering, self).__init__(config)
        self.roberta = RobertaModel(config)
        self.fc = nn.Linear(config.hidden_size * 2, 2)

    def forward(self, input_ids, attention_mask, start_positions=None, end_positions=None):
        _, _, hidden_states = self.roberta(input_ids, attention_mask)
        out = torch.cat([hidden_states[-1], hidden_states[-1]], dim=-1)
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

            loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

            return total_loss

        else:
            return start_logits, end_logits


class BertForQuestionAnswering(BertPreTrainedModel):

    def __init__(self, config):
        super(BertForQuestionAnswering, self).__init__(config)
        self.bert = BertModel(config)
        self.fc = nn.Linear(config.hidden_size * 2, 2)

    def forward(self, input_ids, attention_mask, token_type_ids,
                start_positions=None, end_positions=None):
        _, _, hidden_states = self.bert(input_ids, attention_mask, token_type_ids)
        out = torch.cat([hidden_states[-1], hidden_states[-1]], dim=-1)
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

            loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

            return total_loss

        else:
            return start_logits, end_logits


class AlbertForQuestionAnswering(BertPreTrainedModel):

    def __init__(self, config):
        super(AlbertForQuestionAnswering, self).__init__(config)
        self.albert = AlbertModel(config)
        self.fc = nn.Linear(1024 * 2, 2)

    def forward(self, input_ids, attention_mask, token_type_ids,
                start_positions=None, end_positions=None):
        _, _, hidden_states = self.albert(input_ids, attention_mask, token_type_ids)
        out = torch.cat([hidden_states[-1], hidden_states[-1]], dim=-1)
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

            loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

            return total_loss

        else:
            return start_logits, end_logits


class BertweetForQuestionAnswering(BertPreTrainedModel):

    def __init__(self, config):
        super(BertweetForQuestionAnswering, self).__init__(config)
        self.bertweet = BertLMHeadModel(config)
        self.fc = nn.Linear(768 * 2, 2)

    def forward(self, input_ids, attention_mask, start_positions=None, end_positions=None):
        _, _, hidden_states = self.bertweet(input_ids, attention_mask)
        out = torch.cat([hidden_states[-1], hidden_states[-1]], dim=-1)
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

            loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

            return total_loss

        else:
            return start_logits, end_logits


class QuestionAnswering:

    def __call__(self, model_type):
        if model_type == 'roberta':
            return RobertaForQuestionAnswering
        elif model_type.startswith('bert'):
            if model_type == 'bert':
                return BertForQuestionAnswering
            elif model_type == 'bertweet':
                return BertweetForQuestionAnswering
        elif model_type == 'albert':
            return AlbertForQuestionAnswering

