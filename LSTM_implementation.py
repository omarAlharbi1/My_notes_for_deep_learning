import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets


input_size = 28
sequence_length = 28
num_layers = 4

hidden_size = 150
num_classes = 10
num_epochs = 2
batch_size = 100
learning_rate = 0.001

transf = transforms.ToTensor()
train_dataset = datasets.MNIST(root='./',train=True, transform=transf, download = True)
test_dataset = datasets.MNIST(root='./',train=False, transform=transf, download = True)


train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size,shuffle = True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size,shuffle = True)


class my_RNN(nn.Module):
	def __init__(self, input_size, hidden_size,num_layers, num_classes):
		super(my_RNN, self).__init__()
		self.num_layers = num_layers
		self.hidden_size = hidden_size
		self.lstm = nn.LSTM(input_size,hidden_size,num_layers,batch_first=True)
		self.fc = nn.Linear(hidden_size,num_classes)

	def forward(self,x):
		h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
		c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)

		out,_ = self.lstm(x, (h0,c0))
		# out is a 3D matrix (batches, sequence, features)
		out = out[:, -1, :]

		out = self.fc(out)
		return out

model = my_RNN(input_size,hidden_size,num_layers,num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

n_total_steps = len(train_loader)
for epoch in range(num_epochs):
	for i,(images, labels) in enumerate(train_loader):
		images = images.reshape(-1,sequence_length,input_size)

		outputs = model(images)
		loss = criterion(outputs, labels)

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		if(i+1) % 100 == 0:
			print(f'Epoch [{epoch+1}/{num_epochs}], Steps [{i+1}/{n_total_steps}], Loss:{loss.item():.4f}')

with torch.no_grad():
	n_correct = 0
	n_samples = 0
	for images, labels in test_loader:
		images = images.reshape(-1, sequence_length, input_size)
		outputs = model(images)

		_, predicted = torch.max(outputs.data, 1)
		n_samples+= labels.size(0)
		n_correct += (predicted == labels).sum().item()

	acc = 100.0 * n_correct / n_samples
	print(f'Accuracy: {acc}%')
