import typing as tp

import torch
from torch import nn
from torch.nn import Module
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from src.data.shemas import ConfigData


class View(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, data):
        return data.view(*self.shape)


class MultyModalGPT2Embedder:
    def __init__(self, embedding_layer: nn.Embedding, cfg: ConfigData):
        self.token2emb = embedding_layer
        self.tokenizer = GPT2Tokenizer.from_pretrained(
            cfg.model_cfg.llm_name, pad_token=cfg.model_cfg.pad_token)
        self.img_token = cfg.model_cfg.img_token
        self.state_token = cfg.model_cfg.state_token
        self._add_special_tokens()
        self.img_token_id = self.tokenizer.convert_tokens_to_ids(
            self.img_token)
        self.state_token_id = self.tokenizer.convert_tokens_to_ids(
            self.state_token)
        self.device = cfg.device

    def tokenize(self, text: list[str]) -> torch.Tensor:
        return self.tokenizer(
            text, return_tensors="pt", padding=True
        )["input_ids"].to(self.device)

    def __call__(
        self,
        text: list[str],
        img_embs: tp.Optional[torch.Tensor] = None,
        state_embs: tp.Optional[torch.Tensor] = None
    ) -> tp.Tuple[torch.Tensor, ...]:
        # TODO:
        # - if there no embs but still token in text
        #   remove token or raise a error
        text_token_ids = self.tokenizer(
            text, return_tensors="pt", padding=True
        )["input_ids"].to(self.device)
        img_token_place = (text_token_ids[0] == self.img_token_id).nonzero(
            as_tuple=True)[0]
        state_token_place = (
            text_token_ids[0] == self.state_token_id).nonzero(as_tuple=True)[0]

        # Processing only text in next cases:
        # - no additional embeddings were given
        # - one of embeddings is given, but there were no
        #   specific token in text for embeddings
        # - no special tokens in text were found
        if img_embs is None and state_embs is None or \
                len(img_token_place) == 0 and state_embs is None or \
                len(state_token_place) == 0 and img_embs is None or \
                len(img_token_place) == 0 and len(state_token_place) == 0:
            total_embs = self.token2emb(text_token_ids)
            label_ids = text_token_ids

        # Adding both embeddings to text embeddings in case
        # when both embeddings were given and special tokens
        # were found in text
        elif img_embs is not None and state_embs is not None and \
                len(img_token_place) != 0 and len(state_token_place) != 0:
            if img_token_place > state_token_place:
                total_embs, label_ids = self._concat_two_embs(
                    text_token_ids,
                    state_token_place,
                    img_token_place,
                    state_embs,
                    img_embs
                )
            else:
                total_embs, label_ids = self._concat_two_embs(
                    text_token_ids,
                    img_token_place,
                    state_token_place,
                    img_embs,
                    state_embs
                )

        elif img_embs is not None:
            total_embs, label_ids = self._concat_embs(
                text_token_ids,
                img_embs,
                img_token_place
            )
        elif state_embs is not None:
            total_embs, label_ids = self._concat_embs(
                text_token_ids,
                state_embs,
                state_token_place
            )
        else:
            raise ValueError("Error with logic")

        return (total_embs, label_ids.long())

    def _concat_embs(
        self,
        text_token_ids: torch.Tensor,
        data_embs: torch.Tensor,
        data_place: int
    ) -> tp.Tuple[torch.Tensor, ...]:

        # According to HF docs all ids equal to -100
        # are ignored in loss computation
        data_ignore_token_ids = torch.zeros(
            (text_token_ids.shape[0], data_embs.shape[1]),
            device=self.device
        ) - 100

        left_embs = self.token2emb(text_token_ids[:, :data_place])
        rigth_embs = self.token2emb(text_token_ids[:, data_place + 1:])

        times = left_embs.shape[0] // data_embs.shape[0]
        data_embs = torch.tile(data_embs, (times, 1, 1))
        text_embs_with_data_embs = torch.concat(
            [left_embs, data_embs, rigth_embs], dim=1)

        label_ids = torch.concat([
            text_token_ids[:, :data_place],
            data_ignore_token_ids,
            text_token_ids[:, data_place + 1:]
        ], dim=1)

        return (text_embs_with_data_embs, label_ids)

    def _concat_two_embs(
        self,
        text_token_ids: torch.Tensor,
        first_data_embs_place: int,
        second_data_embs_place: int,
        first_data: torch.Tensor,
        second_data: torch.Tensor,
    ) -> tp.Tuple[torch.Tensor, ...]:

        first_size_ratio = text_token_ids.shape[0] // first_data.shape[0]
        second_size_ratio = text_token_ids.shape[0] // second_data.shape[0]

        first_data = torch.tile(first_data, (first_size_ratio, 1))
        second_data = torch.tile(second_data, (second_size_ratio, 1))

        idx1 = first_data_embs_place
        idx2 = second_data_embs_place

        total_embs = torch.concat([
            self.token2emb(text_token_ids[:, :idx1]),
            first_data,
            self.token2emb(text_token_ids[:, idx1 + 1:idx2]),
            second_data,
            self.token2emb(text_token_ids[:, idx2 + 1:])
        ], dim=1)

        first_data_ignore_token_ids = torch.zeros(
            (text_token_ids.shape[0], first_data.shape[1]),
            device=self.device
        ) - 100
        second_data_ignore_token_ids = torch.zeros(
            (text_token_ids.shape[0], second_data.shape[1]),
            device=self.device
        ) - 100

        label_ids = torch.concat([
            text_token_ids[:, :idx1],
            first_data_ignore_token_ids,
            text_token_ids[:, idx1 + 1:idx2],
            second_data_ignore_token_ids,
            text_token_ids[:, idx2 + 1:]
        ], dim=1)

        return (total_embs, label_ids)

    def _add_special_tokens(self):
        special_tokens_dict = {
            'additional_special_tokens': [
                self.img_token,
                self.state_token
            ]}
        self.tokenizer.add_special_tokens(special_tokens_dict)


class MultyModalGPT2(Module):
    def __init__(self, cfg: ConfigData):
        super().__init__()
        # Initializing LLM model
        self.gpt = GPT2LMHeadModel.from_pretrained(
            cfg.model_cfg.llm_name).to(cfg.device)
        self.token2emb = self.gpt.get_input_embeddings()

        # Initializing multy modal tokenizer
        self.mmtokenizer = MultyModalGPT2Embedder(self.token2emb, cfg)

        # Initializing ResNet features mapper
        feats_size = cfg.model_cfg.imgs_feats_size
        img_tokens_num = feats_size[1]
        img_tokens_dim = feats_size[2]

        self.feats2emb = nn.Sequential(
            nn.Conv2d(cfg.model_cfg.imgs_feats_size[0], 1, 1),
            View((-1, img_tokens_dim)),
            nn.Linear(img_tokens_dim, self.token2emb.embedding_dim),
            View((-1, img_tokens_num, self.token2emb.embedding_dim))
        ).to(cfg.device)

    def forward(self, input_data: dict):
        img_embs = self.feats2emb(input_data["images_features"])

        promt_embs, promt_ids = self.mmtokenizer(
            text=input_data["promts"],
            img_embs=img_embs
        )
        target_embs, target_ids = self.mmtokenizer(
            text=input_data["targets"]
        )

        times = promt_ids.shape[0] // target_ids.shape[0]
        target_embs = torch.tile(target_embs, (times, 1, 1))
        target_ids = torch.tile(target_ids, (times, 1))

        input_embs = torch.concat([promt_embs, target_embs], dim=1)
        label_ids = torch.concat([promt_ids, target_ids], dim=1)

        model_output = self.gpt(
            inputs_embeds=input_embs,
            labels=label_ids
        )

        return model_output
