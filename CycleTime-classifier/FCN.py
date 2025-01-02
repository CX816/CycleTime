import os
import torch
import datetime
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from torchvision.transforms import ToTensor
import time
import copy
from sklearn.metrics import f1_score, recall_score, precision_score

# 记录程序开始时间
start_time = time.time()
Image.MAX_IMAGE_PIXELS = None
class MyDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.transform = ToTensor()
        self.file_list = os.listdir(root_dir)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = os.path.join(self.root_dir, self.file_list[idx])
        image = Image.open(file_path).convert('RGB')
        image = transforms.Resize((24, 24))(image)
        label = int(float(self.file_list[idx].split('_')[2].split('label')[1][:-4]))
        image = self.transform(image)
        return image, label

    def get_samples(self):
        return [image for image, _ in self.data]

    def get_labels(self):
        return [label for _, label in self.data]


class FCN(nn.Module):
    def __init__(self):
        super(FCN, self).__init__()
        # 定义三个隐藏层的卷积层
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3)
        self.pool = nn.MaxPool2d(2, 2)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 64, kernel_size=3)
        self.bn3 = nn.BatchNorm2d(64)

        # 输出层
        self.fc = nn.Linear(64, 2)

    def forward(self, x):
        # 第一个隐藏层
        x = self.conv1(x)
        x = self.pool(x)
        x = self.bn1(x)
        x = F.relu(x)

        # 第二个隐藏层
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)

        # 第三个隐藏层
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool(x)

        x = x.view(x.size(0), -1)

        # 输出层
        x = self.fc(x)
        x = F.softmax(x, dim=1)

        return x


# 定义数据预处理
transform = ToTensor()

# 定义训练集和测试集
train_dataset = MyDataset('train2/')
test_dataset = MyDataset('test2/')
val_dataset = MyDataset('test2/')

# 定义训练集和测试集的 DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)



model = FCN()
# 定义模型、损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# 训练模型
num_epochs = 50


now = datetime.datetime.now()
timestamp = now.strftime("%Y-%m-%d %H:%M:%S")

# 将当前时间保存到文件的第一行
with open('result.txt', 'w') as f:
    f.write('Time of file creation: {}\n'.format(timestamp))
best_val_accuracy = 0.0  # 初始化最好的验证集准确率为0
best_model = None  # 初始化最好的模型为空
for epoch in range(num_epochs):
    running_loss = 0.0
    Loss_list = []
    x_range = []

    for i, data in enumerate(train_loader):
        inputs, labels = data
        #print(inputs.shape)
        optimizer.zero_grad()
        #print(inputs.shape)
        outputs = model(inputs)
        #print(inputs.shape)
        #print(labels.shape)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print('Epoch [%d/%d], Loss: %.4f' % (epoch + 1, num_epochs, running_loss / len(train_loader)))
    model.eval()  # Set the model to evaluation mode

    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        val_accuracy = 0
        for data in val_loader:
            inputs, labels = data

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

        val_accuracy = 100 * val_correct / val_total
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model = copy.deepcopy(model.state_dict())  # 保存当前模型的参数

    val_loss /= len(val_loader)

    print(f'Epoch {epoch + 1}/{num_epochs}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}%')
    # 将loss值和时间戳保存到文件中
    with open('result.txt', 'a') as f:
        f.write('epoch {}: Loss = {},Val Loss = {},Val Acc = {}\n'.format(epoch+1, running_loss / len(train_loader), val_loss, val_accuracy%100))
#print('Accuracy of the network on the Validation images: %2f %%' % best_val_accuracy)
with open('result.txt', 'a') as f:
    f.write('Accuracy of the network on the Validation images:{}'.format(best_val_accuracy))

# 测试模型
correct = 0
total = 0
best_accuracy = 0
predicted_list = []
labels_list = []

# 加载在验证集上表现最好的模型进行测试
# 保存最佳模型参数到文件
torch.save(best_model, 'best_model.pth')

# 加载最佳模型参数
best_model = torch.load('best_model.pth')
model.load_state_dict(best_model)  # 加载在验证集上表现最好的模型的参数
model.eval()  # 将模型设置为评估模式
test_accuracy = 0  # 初始化测试集准确率为0
with torch.no_grad():
    for data in test_loader:
        inputs, labels = data

        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        predicted_list.extend(predicted.cpu().numpy())
        labels_list.extend(labels.cpu().numpy())

    current_accuracy = 100 * correct / total
    f1 = f1_score(labels_list, predicted_list, average='macro')  # 计算F1值
    recall = recall_score(labels_list, predicted_list, average='macro')  # 计算召回率
    precision = precision_score(labels_list, predicted_list, average='macro')  # 计算精确率
print('Accuracy of the network on the test images: %2f %%' % current_accuracy)
print('F1 Score of the network on the test images: %2f' % f1)
print('Recall of the network on the test images: %2f' % recall)
print('Precision of the network on the test images: %2f' % precision)

with open('m.txt', 'a') as f:
    f.write('Accuracy of the network on the test images:{}'.format(current_accuracy))
    f.write('F1 Score of the network on the test images:{}\n'.format(f1))
    f.write('Recall of the network on the test images:{}\n'.format(recall))
    f.write('Precision of the network on the test images:{}\n'.format(precision))

# 记录程序结束时间
end_time = time.time()
total_seconds = end_time - start_time

# 打印和保存运行总时长（秒）
print(f"Total training time: {total_seconds:.2f} seconds")
with open('result.txt', 'a') as f:
    f.write(f"Total training time: {total_seconds:.2f} seconds\n")