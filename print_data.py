# Not really useful
import pandas as pd

# path
file_path = "NASA-Turbofan-data/data/train_FD001.txt"

# columns name
col_names = (['unit', 'time'] +
             [f'setting {i}' for i in range(1, 4)] +       # operating settings
             [f'sensor {i}' for i in range(1, 27)])        # 26 sensors #FIXME I am not sure there is really 26 measurements

# load data
train_FD001 = pd.read_csv(file_path, sep=' ', header=None, names=col_names)

print(train_FD001.head())
print(train_FD001.shape)