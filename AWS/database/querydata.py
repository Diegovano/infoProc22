import boto3
import json
import numpy as np
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.utils.data import TensorDataset
from tqdm import tqdm
from pprint import pprint
from decimal import Decimal 

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
	

def update_points(device_ide,timestamp, dynamodb=None):
	table_name = 'di_data10'
	if not dynamodb:
		dynamodb = boto3.resource('dynamodb', region_name='us-east-1')
	table = dynamodb.Table(table_name)
	response = table.update_item(
		Key={
			'device_id': device_ide,
			'timestamp' : timestamp
		},
		UpdateExpression="set xcordinate=:r, ycordinate=:p",
		ExpressionAttributeValues={
			':r': str(concept_logits[0,0].item()),
			':p': str(concept_logits[0,1].item())
		},
		ReturnValues="UPDATED_NEW"
	)
	
	return response

# device_id to query (so far either 1 or 2) 
dynamodb = boto3.resource('dynamodb', region_name='us-east-1')
table_name = 'DeviceIds'
table = dynamodb.Table(table_name)
response = table.scan()

device_ids = []
for device in response['Items']:
	device_ids.append(device["device_id"])

print(device_ids)
print (device_ids[0])
j = 0
while True:
		device_id = device_ids[j]
		if(j == len(device_ids)-1):
			j =0
		# set up the query parameters
		dynamodb = boto3.client('dynamodb', region_name='us-east-1')
		query_params = {
		'TableName': 'di_data10',
		'KeyConditionExpression': 'device_id = :device_id',
		'ExpressionAttributeValues': {
		':device_id': {'S': device_id},
	},
}
		data = []
	#can add timestamp as well, prob not needed 
		response = dynamodb.query(**query_params)
		prev = 0
		the_model = torch.load("models/model1000epoch10000data_true_decent_3.pth")
		for item in response['Items']:
			print("123123123")
			heading = int(item['heading']['N'])
			total_steps = int(item['total_steps']['N'])
			timestamp = item['timestamp']
			data.append([total_steps, heading])
			if len(data) == 5:
				print("wesrtwertwertqewsrtewrt")
				data_array = np.array(data)
				print(data_array)
				feedin = []
				for i in range(5):
					d_Step = data_array[i][0]-prev
					prev = data_array[i][0]
					feedin.append(d_Step)
					feedin.append(data_array[i][1])
				feedin = np.array(feedin)
				print(feedin)
				feedin = np.expand_dims(feedin,axis=0).astype(float)
				dataa2 = {}
				for i in range(5):
					print("############")
					print(2*i+1)
					print(feedin[:,2*i+1])
					print(np.deg2rad(feedin[:,2*i+1])/(2*np.pi))
					feedin[:,2*i+1] = np.deg2rad(feedin[:,2*i+1])/(2*np.pi)
				print(feedin)
				print(feedin[:,1])
				dataa2['features'] = feedin
				dataa2['labels'] = np.expand_dims(np.array([0,0]),axis=0)
				dataset = toydataset(dataa2)
				a_to_x_test_loader = DataLoader(TensorDataset(torch.tensor([[o[0]] for o in dataset]), torch.tensor([[o[1]] for o in dataset], dtype=torch.float32)), batch_size=BATCH_SIZE, shuffle=False, num_workers=N_WORKERS)
				concept_logits = the_model.predict(a_to_x_test_loader)
				print("PREDICITION==============")
				print(concept_logits)
				print("=================================")
				print(device_ids[j])
		
				print(timestamp['S'])
				update_response = update_points(device_id[j],timestamp['S'])
				print("Update  succeeded:")
				pprint(update_response)
				print(concept_logits)
				prev = data_array[4][0]
				data.clear()
		j += 1

		if len(data) > 0:
			print("Chang you")
			data_array = np.array(data)
			print(data_array)
			
	