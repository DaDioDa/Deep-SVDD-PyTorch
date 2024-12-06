import numpy as np
from collections import Counter
from sklearn.metrics import accuracy_score, f1_score

# 檔案路徑
file_paths = ["VAEpredictions.txt", "AEpredictions.txt"]
ground_truth_file = "True.txt"  # 正確答案

# 讀取所有模型的預測結果
predictions = []
for file_path in file_paths:
    with open(file_path, "r") as f:
        model_predictions = [int(line.strip()) for line in f.readlines()]
        predictions.append(model_predictions)

# 讀取正確答案
with open(ground_truth_file, "r") as f:
    ground_truth = [int(line.strip()) for line in f.readlines()]

# 確保所有模型的預測結果與正確答案長度一致
assert all(len(pred) == len(predictions[0]) for pred in predictions), "預測結果長度不一致"
assert len(ground_truth) == len(predictions[0]), "正確答案與預測結果的樣本數不一致"

# 轉置，方便對每個樣本的預測進行處理
predictions = np.array(predictions).T

# 隨機搜尋參數
num_iterations = 1000  # 隨機搜尋的次數
best_accuracy = 0
best_f1 = 0
best_weights = None

# 隨機搜尋最佳權重
for _ in range(num_iterations):
    # 隨機生成權重並歸一化
    weights = np.random.rand(len(file_paths))
    weights /= np.sum(weights)

    # 加權投票機制
    weighted_predictions = []
    for sample_predictions in predictions:
        weighted_scores = {}
        for i, pred in enumerate(sample_predictions):
            weighted_scores[pred] = weighted_scores.get(pred, 0) + weights[i]
        # 如果平票，預測為 0，否則選加權得分最高的類別
        sorted_scores = sorted(weighted_scores.items(), key=lambda x: (-x[1], x[0]))
        if len(sorted_scores) > 1 and sorted_scores[0][1] == sorted_scores[1][1]:
            weighted_predictions.append(0)
        else:
            weighted_predictions.append(sorted_scores[0][0])

    # 計算 Accuracy 和 F1-Score
    accuracy = accuracy_score(ground_truth, weighted_predictions)
    f1 = f1_score(ground_truth, weighted_predictions)

    # 更新最佳結果
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_f1 = f1
        best_weights = weights

# 輸出最佳結果
print(f"最佳權重: {best_weights}")
print(f"最佳 Accuracy: {best_accuracy:.4f}")
print(f"最佳 F1-Score: {best_f1:.4f}")
