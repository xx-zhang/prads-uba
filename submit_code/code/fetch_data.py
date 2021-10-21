# coding:utf-8
import os
APP_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
train_data = pd.read_csv(os.path.join(APP_DIR, "test_data", "train_data.csv"), encoding="gbk")

print(train_data.head(5))
