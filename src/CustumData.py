import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from imblearn.combine import SMOTEENN
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, data, labels):
        """
        初始化 Dataset
        :param data: 已處理的特徵數據 (NumPy array or Pandas DataFrame)
        :param labels: 對應的標籤數據
        """
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        """返回數據集大小"""
        return len(self.data)

    def __getitem__(self, idx):
        """根據索引返回一筆數據及其標籤"""
        return self.data[idx], self.labels[idx]

def process_data(file_path):
    # 讀取資料
    df = pd.read_csv(file_path)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    df = df.drop('id', axis=1)
    
    # 標籤處理
    y = df['marker']
    y = [0 if value in [41] else 1 for value in y]
    X = df.drop('marker', axis=1)

    # 刪除不必要的欄位
    columns_to_drop = ['control_panel_log1', 'control_panel_log2', 'control_panel_log3', 
                       'control_panel_log4', 'relay1_log', 'relay2_log', 'relay3_log', 
                       'relay4_log', 'snort_log1', 'snort_log2', 'snort_log3', 'snort_log4']
    X = X.drop(columns=columns_to_drop, axis=1)
    
    # 分割數據集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # PCA 降維
    pca = PCA(n_components=15)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)

    # SMOTE-ENN 平衡數據
    smote_enn = SMOTEENN(random_state=0)
    X_train, y_train = smote_enn.fit_resample(X_train, y_train)

    # Min-Max 標準化
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test

def create_dataloaders(file_path, batch_size=32, num_workers=0):
    # 處理數據
    X_train, X_test, y_train, y_test = process_data(file_path)

    # 建立 Dataset
    train_dataset = CustomDataset(X_train, y_train)
    test_dataset = CustomDataset(X_test, y_test)

    # 建立 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader

# # 檔案路徑
# file_path = r'multiclass15\csv_result-data12 Sampled Scenarios.csv'

# # 創建 DataLoader
# train_loader, test_loader = create_dataloaders(file_path, batch_size=64)

