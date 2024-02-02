import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset

# Define custom dataset class
class CustomDataset(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        embedding_numpy = np.fromstring(self.data.iloc[index]['embedding_code'][1:-1], sep=' ', dtype=np.float32)
        embedding_code = torch.tensor(embedding_numpy)
        label = torch.tensor(float(self.data.iloc[index]['label']))
        return embedding_code, label
    
# Define classification network
class WebshellDetector(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super(WebshellDetector, self).__init__(*args, **kwargs)
        self.conv1 = torch.nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3)
        self.maxpool = torch.nn.MaxPool1d(kernel_size=3)
        self.conv2 = torch.nn.Conv1d(in_channels=32, out_channels=16, kernel_size=3)
        self.avgpool = torch.nn.AvgPool1d(kernel_size=2)
        self.dropout = torch.nn.Dropout(p=0.5)
        self.fc1 = torch.nn.Linear(320, 128)
        self.fc2 = torch.nn.Linear(128, 64)
        self.fc3 = torch.nn.Linear(64, 1)
        self.sigmoid = torch.nn.Sigmoid()
    
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.sigmoid(x).squeeze()
        return x

def train_model(model, criterion, optimizer, dataloader, num_epochs):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in tqdm(dataloader, desc=f'Epoch {epoch + 1}/{num_epochs}'):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(dataloader)}')

def inference(model, dataloader, threshold):
    model.eval()
    all_predictions = []
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc='Inference'):
            outputs = model(inputs)
            predictions = (outputs > threshold).float()
            all_predictions.extend(predictions.squeeze().tolist())
    return all_predictions
