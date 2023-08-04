""" Dataset implementation module """

import json
import os

import torch
from torch.utils.data import Dataset

from src.data.shemas import ConfigData


class AlfredDataset(Dataset):
    def __init__(
        self,
        cfg: ConfigData
    ):
        data_folder_path = cfg.train_cfg.train_data_path

        self.data_path = data_folder_path
        self.img_token = cfg.model_cfg.img_token
        self.state_token = cfg.model_cfg.state_token

        self.data_folders_names = [
            folder + "/" + subfolder + "/"
            for folder in os.listdir(data_folder_path)
            for subfolder in os.listdir(data_folder_path+folder)
        ]

    def _get_data_sample(self, path: str):
        features_path = path + "feat_conv.pt"
        traj_data_path = path + "traj_data.json"

        features = torch.load(features_path)[:-1]
        with open(traj_data_path) as filep:
            traj_data = json.load(filep)

        tasks_num = len(traj_data["turk_annotations"]["anns"])
        anns = traj_data["turk_annotations"]["anns"]
        instructions = []

        for task_idx in range(tasks_num):
            goal = anns[task_idx]['task_desc'].lower(
            ).strip().replace('\n', '')
            high_descs = [''.join(ch for ch in desc).lower().
                          strip().replace('\n', '') for desc in
                          anns[task_idx]['high_descs']]
            instructions.append((goal + ' ' + ' '.join(high_descs)).lower())

        low_actions = []
        for low_action in traj_data["plan"]["low_actions"]:
            low_actions += [low_action["api_action"]["action"].lower()]

        output = {
            "instructions": instructions,
            "actions": low_actions,
            "images_features": features
        }
        return output

    def _promt_generator(self, data_sample: dict):
        promts = []

        for instruction_idx in range(len(data_sample["instructions"])):
            for i, _ in enumerate(data_sample["actions"]):
                instruction = data_sample["instructions"][instruction_idx]
                prev_actions = ". ".join(data_sample["actions"][:i])
                promt = f"You see: {self.img_token}. Task: {instruction}. \
                          Your previous actions: {prev_actions}. \
                          What is you next step?"
                promts.append(promt)

        return {
            "promts": promts,
            "images_features": data_sample["images_features"],
            "targets": data_sample["actions"]
        }

    def __getitem__(self, index) -> dict:
        sample_path = self.data_path + self.data_folders_names[index]
        data_sample = self._get_data_sample(sample_path)
        promt = self._promt_generator(data_sample)
        return promt

    def __len__(self):
        return len(self.data_folders_names)
