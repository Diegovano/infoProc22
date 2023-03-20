import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.utils.data import TensorDataset
from sklearn.linear_model import LogisticRegression
import steptoy as toy
from tqdm import tqdm
import matplotlib as plot

DEVICE = 'cuda'
BATCH_SIZE = 1024
N_WORKERS = 0

N_CONCEPTS = 500

HPARAMS = ((0.001, 10), 
		   (0.001, 200))
seedyseed = 1412

class toydataset(Dataset):
	def __init__(self, generated):
		self.features = generated['features'][:]
		self.labels = generated['labels'][:]
		self.features = self.features.astype('float32')
		self.labels = self.labels.astype('float32')
	
	def __len__(self):
		return len(self.features)
	
	def __getitem__(self, idx):
		return [self.features[idx], self.labels[idx]]

	def get_splits(self, n_test=0.33):
		# determine sizes
		test_size = round(n_test * len(self.features))
		train_size = len(self.features) - test_size
		# calculate the split
		return random_split(self, [train_size, test_size])


class Model(nn.Module):
	def __init__(self, layer_sizes=[2, 16, 16, 1], activation=nn.ReLU,
				 final_activation=None, lr=0.001,  loss=nn.functional.cross_entropy,
				 device='cuda', optim_kwargs={}):
		
		super().__init__()
		layers = []
		for i in range(len(layer_sizes) - 1):
			layers.extend([
				nn.Linear(layer_sizes[i], layer_sizes[i+1]),
				activation(),
			])
		
		# Pop off the last activation
		layers.pop()
		if final_activation:
			layers.append(final_activation)

		self.model = nn.Sequential(*layers)
		self.loss = loss
		self.device = device
		self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, **optim_kwargs)
		
		
	
	def forward(self, x):
		return self.model.forward(x)

	def predict(self, dataloader):
		self.eval()
		outputs = []
		with torch.no_grad():
			for data in dataloader:
				if (isinstance(data, tuple) or isinstance(data, list)) and len(data) > 1:
					inputs = data[0]
				else:
					inputs = data
				#inputs = inputs.to(self.device)
				outputs.extend(self(inputs))
		return torch.vstack(outputs)

	def fit(self, trainloader, testloader=None, report_every=5, n_epoch=30):
		self.train()
		losses = {}
		test_losses = {}
		for epoch in tqdm(range(n_epoch)):  # loop over the dataset multiple times
			running_loss = 0.0
			for i, data in enumerate(trainloader):
				inputs, labels = data
				#inputs = inputs.to(self.device)
				#labels = labels.to(self.device)
				#print(torch.cuda.device_count())
				#print(torch.cuda.is_available())

				# zero the parameter gradients
				
				self.optimizer.zero_grad()
				# forward + backward + optimize
				outputs = self(inputs)
				
				loss = self.loss(outputs, labels)
				losses[epoch*len(trainloader)+i] = loss.mean().item()
				#print(outputs, labels, loss)

				loss.backward()
				self.optimizer.step()
				print(loss)
				
				# Skip the rest if there's no validation set.
				if testloader is None:
					continue
	
				self.eval()
				with torch.no_grad():
					for j, data in enumerate(testloader, 0):
						# get the inputs; data is a list of [inputs, labels]
						inputs, labels = data
						#inputs = inputs.to(self.device)
						#labels = labels.to(self.device)

						outputs = self(inputs)
						loss = self.loss(outputs, labels)
						test_losses[epoch*len(trainloader)+i] = loss.mean().item()
				self.train()
			#print(epoch)

			if testloader is None:
				continue

		return (losses, test_losses)
	
def obs_to_concepts(o):
	return o[1]

def a_to_x(train, test, seed, hparams):
	
	a_to_x_train_loader = DataLoader(TensorDataset(torch.tensor([[o[0]] for o in train]), torch.tensor([[o[1]] for o in train], dtype=torch.float32)), batch_size=BATCH_SIZE, shuffle=True, num_workers=N_WORKERS)
	a_to_x_test_loader = DataLoader(TensorDataset(torch.tensor([[o[0]] for o in test]), torch.tensor([[o[1]] for o in test], dtype=torch.float32)), batch_size=BATCH_SIZE, shuffle=False, num_workers=N_WORKERS)

	a_to_x_model = Model(layer_sizes=[10,1024,1024,128,2], 
				  activation=nn.ReLU,
				  final_activation=None,
				  lr=hparams[0],
				  loss=nn.functional.l1_loss, device='cuda')

	train_loss, test_loss = a_to_x_model.fit(a_to_x_train_loader, None, n_epoch=hparams[1])

	concept_labels = [obs_to_concepts(o) for o in test]
	concept_logits = a_to_x_model.predict(a_to_x_test_loader).cpu()
	concept_preds = torch.sigmoid(concept_logits) > 0.5
	print("=============================")
	result = np.c_[[c[0] for c in concept_labels],[c[1] for c in concept_labels], concept_logits]
	print(np.c_[[c[0] for c in concept_labels],[c[1] for c in concept_labels], concept_logits])
	print("=============================")
	#print(f'Concept 0 acc: {accuracy_score([c[0] for c in concept_labels], concept_preds[:,0].cpu())}')
	#print(f'Concept 1 acc: {accuracy_score([c[1] for c in concept_labels], concept_preds[:,1].cpu())}')
	# print(f'Concept 2 acc: {accuracy_score([c[2] for c in concept_labels], concept_preds[:,2].cpu())}')
	#pd.DataFrame({'train': train_loss, 'test': test_loss}).plot(
	#	title=f'Features to Concepts - {n_concepts}, {seed}')
	#plt.show()
	
	return a_to_x_model, result

dataa = toy.datagen(10000,5, True, 0.01)
dataset = toydataset(dataa)
train, test = dataset.get_splits()
HPARAMS = ((0.01, 1000), 
		   (0.001, 200))

modl,result = a_to_x(train, test, 1111, HPARAMS[0])


torch.save(modl, "C:\\Users\\eddyc\\Documents\\IP\\models\\model1000epoch10000data.pth")

a_to_x_test_loader = DataLoader(TensorDataset(torch.tensor([[o[0]] for o in test]), torch.tensor([[o[1]] for o in test], dtype=torch.float32)), batch_size=BATCH_SIZE, shuffle=False, num_workers=N_WORKERS)

the_model = torch.load("C:\\Users\\eddyc\\Documents\\IP\\models\\model1000epoch10000data.pth")

concept_logits = the_model.predict(a_to_x_test_loader)
print(concept_logits)