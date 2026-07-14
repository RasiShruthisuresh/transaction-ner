"""Token classification model: architecture + config.

Config is a dataclass, not hardcoded constants, so Phase 6 upgrade paths
(a bigger/different encoder) are a one-line change + rerun, not a rewrite.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

from config import ALL_TAGS


@dataclass
class ModelConfig:
    model_name: str = "jhu-clsp/ettin-encoder-32m"
    max_length: int = 64
    dropout: float = 0.1
    num_labels: int = len(ALL_TAGS)


class TokenClassifier(nn.Module):
    """Generic encoder + linear token-classification head.

    Built as a plain AutoModel + Linear rather than
    AutoModelForTokenClassification: the baseline encoder
    (jhu-clsp/ettin-encoder-32m) and some Phase 6 candidates aren't
    guaranteed to have a registered *ForTokenClassification class in
    every transformers version, but AutoModel + a manual head works with
    anything AutoModel can load -- which is what actually needs to stay
    swappable across encoders.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.encoder = AutoModel.from_pretrained(config.model_name)
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(hidden_size, config.num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        seq_out = self.dropout(out.last_hidden_state)
        logits = self.classifier(seq_out)
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        return {"loss": loss, "logits": logits}


def build_tokenizer(config: ModelConfig):
    return AutoTokenizer.from_pretrained(config.model_name)


def build_model(config: ModelConfig) -> TokenClassifier:
    return TokenClassifier(config)
