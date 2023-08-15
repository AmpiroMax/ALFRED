""" Dataset implementation module """

import json
import os

import torch
from torch.utils.data import Dataset

from src.data.shemas import ConfigData


PROMT_TYPES = [
    "task2low_actions",
    "instruction2low_actions",
    "instruction_img2next_step",
]


class AlfredDataset(Dataset):
    def __init__(
        self,
        cfg: ConfigData,
        dataset_type: str = "train",
        promt_type: str = "instruction_img2next_step"
    ):
        if dataset_type == "train":
            data_folder_path = cfg.train_cfg.train_data_path
        elif dataset_type == "test":
            data_folder_path = cfg.train_cfg.test_data_path
        else:
            raise ValueError("Wrong dataset type was given")

        self.promt_type = promt_type

        self.data_path = data_folder_path
        self.img_token = cfg.model_cfg.img_token
        self.state_token = cfg.model_cfg.state_token
        self.bos_token = cfg.model_cfg.bos_token
        self.eos_token = cfg.model_cfg.eos_token

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
        tasks = []
        for task_idx in range(tasks_num):
            goal = anns[task_idx]['task_desc'].lower(
            ).strip().replace('\n', '')
            high_descs = [''.join(ch for ch in desc).lower().
                          strip().replace('\n', '') for desc in
                          anns[task_idx]['high_descs']]
            instructions.append((goal + ' ' + ' '.join(high_descs)).lower())
            tasks.append(goal)
        low_actions = []
        for low_action in traj_data["plan"]["low_actions"]:
            low_actions.append(low_action["api_action"]["action"].lower())

        output = {
            "tasks": tasks,
            "instructions": instructions,
            "actions": low_actions,
            "images_features": features
        }
        return output

    def _promt_task2low_actions(self, data_sample: dict):
        promts = []
        targets = []

        for task_idx in range(len(data_sample["tasks"])):
            task = data_sample["tasks"][task_idx]
            promt = f"{self.bos_token}Task: {task}. What is you action plan?{self.eos_token}"
            promts.append(promt)
            targets.append(" ".join(data_sample["actions"]))

        return {
            "promts": promts,
            "images_features": None,
            "targets": targets
        }

    def _promt_instruction2low_actions(self, data_sample: dict):
        promts = []
        targets = []

        for instruction_idx in range(len(data_sample["instructions"])):
            instruction = data_sample["instructions"][instruction_idx]
            promt = f"{self.bos_token}Instruction: {instruction}. What is you action plan?{self.eos_token}"
            promts.append(promt)
            targets.append(" ".join(data_sample["actions"]))

        return {
            "promts": promts,
            "images_features": None,
            "targets": targets
        }

    def _promt_instruction_img2next_step(self, data_sample: dict):
        promts = []

        for instruction_idx in range(len(data_sample["instructions"])):
            for i, _ in enumerate(data_sample["actions"]):
                instruction = data_sample["instructions"][instruction_idx]
                prev_actions = ". ".join(data_sample["actions"][:i])
                promt = f"{self.bos_token}You see: {self.img_token}. Task: {instruction}. Your previous actions: {prev_actions}. What is you next step?{self.eos_token}"
                promts.append(promt)

        return {
            "promts": promts,
            "images_features": data_sample["images_features"],
            "targets": data_sample["actions"]
        }

    def __getitem__(self, index) -> dict:
        sample_path = self.data_path + self.data_folders_names[index]
        data_sample = self._get_data_sample(sample_path)
        promt = dict()
        if self.promt_type == "task2low_actions":
            promt = self._promt_task2low_actions(data_sample)
        elif self.promt_type == "instruction2low_actions":
            promt = self._promt_instruction2low_actions(data_sample)
        elif self.promt_type == "instruction_img2next_step":
            promt = self. _promt_instruction_img2next_step(data_sample)
        return promt

    def __len__(self):
        return len(self.data_folders_names)
