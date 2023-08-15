from typing import List, Optional

from pydantic import BaseModel, Field


class ConfigModel(BaseModel):
    llm_name: str = "gpt2"
    vit_name: str = ""
    n_visual_tokens: int = 1
    bos_token: str = "<|endoftext|>"
    eos_token: str = "<|endoftext|>"
    pad_token: str = "<|endoftext|>"
    sep_token: str = "<|endoftext|>"
    img_token: str = "<|image|>"
    state_token: str = "<|state|>"
    imgs_feats_size: tuple = (512, 7, 7)


class ConfigTraining(BaseModel):
    train_data_path: str = "../data"
    test_data_path: str = "../data"
    data_samples_num: int = Field(default=-1, type=int)
    epoch_num: int = Field(default=1, gt=0, type=int)
    lr: float = Field(default=1e-2, gt=0, type=float)
    lr_decay: float = Field(default=0.99, gt=0, le=1, type=float)
    batch_size: int = Field(default=500, gt=1, type=int)
    grad_clip: int = Field(default=50, type=int)


class ConfigData(BaseModel):
    model_load_path: Optional[str] = None
    model_save_path: str = '../models/'
    device: str = "cpu"
    model_cfg: ConfigModel = ConfigModel()
    train_cfg: ConfigTraining = ConfigTraining()
