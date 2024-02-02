import os
import time
import torch
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from config import *
from utils import *
from process_code import process_train_folder

# Load existed processed samples or initialize processing progress
if os.path.exists(output_samples_csv_folder) and os.listdir(output_samples_csv_folder):
    files = os.listdir(output_samples_csv_folder)
    recent_file = max(files, key=lambda f: os.path.getmtime(os.path.join(output_samples_csv_folder, f)))
    samples_df = pd.read_csv(os.path.join(output_samples_csv_folder, recent_file))
else:
    samples_df = process_train_folder()

samples_df = samples_df.sample(frac=1).reset_index(drop=True)
train_df, test_df = train_test_split(samples_df, test_size=0.2, random_state=23)

train_dataset = CustomDataset(train_df)
test_dataset = CustomDataset(test_df)

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)

webshell_detector = WebshellDetector()
criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(webshell_detector.parameters(), lr=0.005)

train_model(webshell_detector, criterion, optimizer, train_dataloader, num_epochs=30)
torch.save(webshell_detector.state_dict(), os.path.join(classification_model_path, f'webshell_detector_model_{str(time.time())}.pth'))

loaded_model = WebshellDetector()
models = os.listdir(classification_model_path)
recent_model = max(models, key=lambda f: os.path.getmtime(os.path.join(classification_model_path, f)))
loaded_model.load_state_dict(torch.load(os.path.join(classification_model_path, recent_model)))

test_predictions = inference(loaded_model, test_dataloader, threshold=0.5)
test_labels = test_df['label'].tolist()
evaluation_result = classification_report(test_labels, test_predictions)
print(f'Test Report:\n{evaluation_result}')
