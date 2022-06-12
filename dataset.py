import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset
import sqlite3
import torch
import time

"""
250, 3373700
500, 6896900
1500, 20375200
5000, 67548200
"""


class SSLDataset(Dataset):
    def __init__(self, table_name, device, batch_size):
        self.con = sqlite3.connect("replays-waddles.db")
        cursor = self.con.cursor()
        self.table_name = table_name
        self.device = device
        helper_data = cursor.execute(f"SELECT * FROM {table_name}")
        cursor.close()
        self.column_names = list(map(lambda x: x[0], helper_data.description))  # get the names of columns
        self.columns = ", ".join(
            name for name in self.column_names if (name != "id" and name != "file_id"))  # join the relevant columns
        self.batches_served = 0
        self.batchsize = batch_size

    def shuffle_whole_dataset(self):
        cursor = self.con.cursor()
        cursor.execute(f"DROP TABLE random{self.table_name}")
        cursor.close()
        cursor = self.con.cursor()
        self.data = cursor.execute(
            f"""CREATE TABLE random{self.table_name} AS SELECT {self.columns} FROM {self.table_name} WHERE {' AND '.join(column + ' IS NOT NULL' for column in self.column_names)} ORDER BY RANDOM()""")
        cursor.close()
        print("Shuffled table")

    def __len__(self):
        cursor = self.con.cursor()
        l = cursor.execute(f"SELECT COUNT(*) FROM random{self.table_name}").fetchone()[0]
        cursor.close()
        return l

    def get_batch(self):
        expression =f'''select {self.columns} from {self.table_name} WHERE rowid in (select distinct (1+abs(random()) % (SELECT rowid FROM {self.table_name} ORDER BY rowid DESC LIMIT 1)) from {self.table_name} WHERE {" AND ".join("("+name+" IS NOT NULL)" for name in self.column_names if (name != "id" and name != "file_id"))} limit {self.batchsize});'''
        cursor = self.con.cursor()
        cursor.execute(expression)
        inp = cursor.fetchall()
        cursor.close()
        self.batches_served += 1
        a = torch.tensor(inp, device=self.device)
        return a


if __name__ == '__main__':
    dataset = SSLDataset("ssl_2v2_validation", device=torch.device("cuda"),batch_size=100_000)
    dataset.shuffle_whole_dataset()
    print(dataset.columns, len(dataset.column_names))
