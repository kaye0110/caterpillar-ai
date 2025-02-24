import torch
from torch.utils.data import Dataset


class StockDataset(Dataset):
    def __init__(self, data, time_step, predict_days):
        self.data = data
        self.time_step = time_step
        self.predict_days = predict_days

    def __len__(self):
        return len(self.data) - self.time_step - self.predict_days + 1

    def __getitem__(self, index):
        X = self.data[index:(index + self.time_step), :-1]
        y = self.data[(index + self.time_step):(index + self.time_step + self.predict_days), -1]
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
