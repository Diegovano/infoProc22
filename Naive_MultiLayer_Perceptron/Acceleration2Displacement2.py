import torch
from torch import nn
# pytorch mlp for binary classification
import matplotlib.pyplot as plt

import pandas as pd
from numpy import vstack
from pandas import read_csv
from sklearn import preprocessing
from sklearn import metrics
#from sklearn.preprocessing import LabelEncoder
#from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.utils.data import TensorDataset
from torch import Tensor
from torch.nn import Linear
from torch.nn import Sigmoid
from torch.nn import Module
from torch.nn import BCELoss
from torch.optim import SGD
import torch

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
DEVICE = 'cuda'
BATCH_SIZE = 1024
N_WORKERS = 0

N_CONCEPTS = 500

HPARAMS = ((0.001, 10), 
		   (0.001, 200))

collate_x_to_c = lambda ds: (torch.tensor([d.image.flatten() for d in ds], dtype=torch.float32), 
							   torch.tensor([obs_to_concepts(d) for d in ds], dtype=torch.float32))

seedyseed = 4568
def generate_dataset( random_seed ):

	np.random.seed( random_seed )

	n_instances = 10000

	n_correlated_groups = 6
	alpha = 1.5
	beta = 2.7

	n_features_per_correlated_group = 2
	n_random_features = 12
	p_flip_bits = 0.05

	concepts_for_correlated_groups = [ 0 , 0 , 1 , 1, 2, 2 ]

	def l_than(t):
		return lambda x: 1*(x < t)
	
	def g_than(t):
		return lambda x: 1*(x > t)

	
	feature_thresholds = [g_than(0.40), g_than(0.60), l_than(0.20), l_than(0.60), # concept 0
						  l_than(0.20), l_than(0.40), g_than(0.60), g_than(0.60), # concept 1
						  g_than(0.20), g_than(0.80), l_than(0.80), l_than(0.20), # concept 2
						  g_than(0.40), g_than(0.60), l_than(0.20), l_than(0.60), # irrelevant
						  l_than(0.20), l_than(0.40), g_than(0.60), g_than(0.60), # irrelevant
						  g_than(0.20), g_than(0.80), l_than(0.80), l_than(0.20)] # irrelevant
	

	correlated_group_features = np.random.beta(alpha, beta, size=( n_instances , n_correlated_groups ) )

	concept_feature_ids = []
	correlated_group_ids = []
	concept_ids = []
	features = []
	for i in range( n_correlated_groups ):
		for j in range( n_features_per_correlated_group ):
			flip_bits = np.random.binomial( n=1 , p=p_flip_bits , size=( n_instances ) ) ##n=1 trial

			feature_column = correlated_group_features[:, i].copy()
			feature_column[np.where(flip_bits)] = 1 - feature_column[np.where(flip_bits)]

			features.append( feature_column )
			correlated_group_ids.append( i )
			concept_ids.append( concepts_for_correlated_groups[ i ] )


	for i in range( n_random_features ):
		feature_column = np.random.beta( alpha, beta, size=( n_instances ) ) 
		features.append( feature_column )
		correlated_group_ids.append( -1 )
		concept_ids.append( -1 )
 
	concept_ids = np.array( concept_ids )

	features = np.swapaxes( np.array( features ) , 0 , 1 )

	# binary features
	features_binarized = np.zeros(features.shape)
	for i in range(n_correlated_groups*n_features_per_correlated_group+n_random_features):
		features_binarized[:, i] = feature_thresholds[i](features[:, i])


	concepts = np.zeros( ( n_instances , 3 ) )
	for concept_id in range( 3 ):
		correlated_group_ids_for_concept = np.where( concepts_for_correlated_groups == concept_id )
		feature_ids_for_concept = np.where( concept_ids == concept_id )[ 0 ]
	
		concept_features_sum = np.sum( features_binarized[ : , feature_ids_for_concept ] , axis=1 )

		concepts[ : , concept_id ][ np.where( concept_features_sum > 0 )[ 0 ] ] = 1


	feature_to_concept_rules = {}
	feature_to_concept_rules[ 'rule_type' ] = 'and' 
	feature_to_concept_rules[ 'labels' ] = [ i for i in range( 3 ) ]
	feature_to_concept_rules[ 'rule_order' ] = [ i for i in range( 3 ) ]
	feature_to_concept_rules[ 'default_label' ] = -1 ##For now, set default label to -1, but shouldn't matter because you don't need to have any!
	feature_to_concept_rules[ 'predicate_counts' ] = [ [ 1 ] , [ 1 ] , [ 1 ] ]
	feature_to_concept_rules[ 'predicate_lists' ] = [ [ np.where( concept_ids == concept_id )[ 0 ].tolist() ] for concept_id in range( 3 ) ]

	lr = LogisticRegression( penalty='none' ) #penalty='l2' , C=1. )

	##Fit to sth random so intercept and coefs get initialized and I can manually set them
	lr.fit( concepts , np.random.binomial( n=1 , p=0.5 , size=( len( features ) ) ) )

	lr.intercept_[ 0 ] = 0.
	lr.coef_ = np.array( [ [ -0.6 , 0.4 , 0.4 ] ] )

	labels = lr.predict( concepts )

	##Assert the feature to concept rules and concept to label model actually generate labels!
	# preds = model.predict_optimized( features_binarized , feature_to_concept_rules , lr , 'logistic_regression' )
	# assert accuracy_score( preds , labels ) , "Re-creating labels isn't working!! " + str( accuracy_score( preds , labels ) )

	
	n_train = int( 0.6 * features.shape[ 0 ] )
	n_val = int( 0.2 * features.shape[ 0 ] )
	train_indices = np.random.choice( np.arange( features.shape[ 0 ] ) , n_train , replace=False )
	val_indices = np.random.choice( np.setdiff1d( np.arange( features.shape[ 0 ] ) , train_indices ) , n_val , replace=False )
	test_indices = np.setdiff1d( np.arange( features.shape[ 0 ] ) , np.array( train_indices.tolist() + val_indices.tolist() ) )
	np.random.shuffle( test_indices )

	assert len( np.intersect1d( val_indices , train_indices ) ) == 0.
	assert len( np.intersect1d( test_indices , val_indices ) ) == 0.
	assert len( np.intersect1d( test_indices , train_indices ) ) == 0.
	assert len( train_indices ) + len( val_indices ) + len( test_indices ) == features.shape[ 0 ]

	dataset = {}
	dataset[ 'features' ] = features_binarized
	dataset['features_continuous'] = features
	dataset[ 'labels' ] = labels
	dataset[ 'onehot_labels' ] = np.zeros( ( n_instances , 2 ) )
	dataset[ 'onehot_labels' ][ np.arange( len( labels ) ) , labels ] = 1
	dataset[ 'concepts' ] = concepts
	dataset[ 'concept_names' ] = [ 'concept'+str( i ) for i in range( concepts.shape[ 1 ] ) ]
	dataset[ 'feature_names' ] = [ 'feature'+str( i ) for i in range( features_binarized.shape[ 1 ] ) ]
	dataset[ 'concept_to_label_model' ] = lr
	dataset[ 'feature_to_concept_rules' ] = feature_to_concept_rules
	dataset[ 'train_indices' ] = train_indices
	dataset[ 'valid_indices' ] = val_indices
	dataset[ 'test_indices' ] = test_indices

	return dataset

class toydataset(Dataset):
	def __init__(self, generated):
		#self.features = generated['features'][:]
		self.features_continuous = generated['features_continuous'][:]
		self.concepts = generated['concepts'][:]
		self.labels = generated['labels'][:]
		self.features_continuous = self.features_continuous.astype('float32')
		# label encode target and ensure the values are floats
		#self.concepts = preprocessing.LabelEncoder().fit_transform(self.concepts)
		self.concepts = self.concepts.astype('float32')
		self.labels = self.labels.astype('float32')
		#self.concepts = self.concepts.reshape((len(self.concepts), 1))
	
	def __len__(self):
		return len(self.features_continuous)
	
	def __getitem__(self, idx):
		return [self.features_continuous[idx], self.concepts[idx], self.labels[idx]]
		#return [self.features_continuous[idx], self.concepts[idx]]

	def get_splits(self, n_test=0.33):
		# determine sizes
		test_size = round(n_test * len(self.features_continuous))
		train_size = len(self.features_continuous) - test_size
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
		for epoch in range(n_epoch):  # loop over the dataset multiple times
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
			print(epoch)

			if testloader is None:
				continue

		return (losses, test_losses)

class CBModel(nn.Module):
	def __init__(self, layer_sizes=[[2, 16, 16, 16, 2], [16, 1]], lr=0.001, concept_loss_weight=1.0):
		super().__init__()
		self.concept_loss_weight = concept_loss_weight
		concept_layers = []
		for i in range(len(layer_sizes[0]) - 1):
			concept_layers.extend([
				nn.Linear(layer_sizes[0][i], layer_sizes[0][i+1]),
				nn.ELU(),
			])
		concept_layers.pop()

		task_layers = [nn.ELU(), nn.Linear(layer_sizes[0][-1], layer_sizes[1][0]), nn.ELU()]
		for i in range(len(layer_sizes[1]) - 1):
			task_layers.extend([
				nn.Linear(layer_sizes[1][i], layer_sizes[1][i+1]),
				nn.ELU(),
			])
		task_layers.pop()
		task_layers.append(nn.Sigmoid())

		self.concept_encoder = nn.Sequential(*concept_layers)
		self.task_model = nn.Sequential(*task_layers)
		self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
	
	def forward(self, x):
		concepts = self.concept_encoder(x)
		output = self.task_model(concepts)
		return concepts, output

	def loss(self, concept_preds, concepts, preds, labels):
		concept_loss = nn.functional.mse_loss(concept_preds, concepts)
		task_loss = nn.functional.binary_cross_entropy(preds, labels)
		return (concept_loss * self.concept_loss_weight + task_loss,
				concept_loss,
				task_loss)

	def predict(self, dataloader):
		self.eval()
		preds = []
		with torch.no_grad():
			for data in dataloader:
				if isinstance(data, tuple) and len(data) > 1:
					inputs = data[0]
				else:
					inputs = data
				preds.extend(self(inputs)[1])
		return torch.vstack(preds)

	def fit(self, trainloader, testloader=None, report_every=5, n_epoch=30):
		self.train()
		losses = {}
		test_losses = {}
		for epoch in range(n_epoch):  # loop over the dataset multiple times
			running_loss = 0.0
			for i, data in enumerate(trainloader, 0):
				# get the inputs; data is a list of [inputs, labels]
				inputs, concepts, labels = data

				# zero the parameter gradients
				self.optimizer.zero_grad()

				# forward + backward + optimize
				concept_preds, outputs = self(inputs)
				loss, concept_loss, task_loss = self.loss(concept_preds, concepts, outputs, labels)
				losses[epoch*len(trainloader)+i] = ((loss.mean().item(), concept_loss.mean().item(), task_loss.mean().item()))

				loss.backward()
				self.optimizer.step()

				# print statistics
				running_loss += loss.item()
				if i % report_every == report_every-1:
					print('[%d, %5d] loss: %.3f' %
						(epoch + 1, i + 1, running_loss / report_every))
					running_loss = 0.0

				# Skip the rest if there's no validation set.
				if testloader is None:
					continue

				self.eval()
				with torch.no_grad():
					for j, data in enumerate(testloader, 0):
						# get the inputs; data is a list of [inputs, labels]
						inputs, concepts, labels = data

						concept_preds, outputs = self(inputs)
						loss, concept_loss, task_loss = self.loss(concept_preds, concepts, outputs, labels)
						test_losses[epoch*len(trainloader)+i] = ((loss.mean().item(), concept_loss.mean().item(), task_loss.mean().item()))
				self.train()
			
		return (losses, test_losses)
		
def obs_to_concepts(o):
	return o[1]
	
def x_to_c(train, test, seed, hparams):
	n_concepts = len(obs_to_concepts(train[0]))
	
	x_to_c_train_loader = DataLoader(TensorDataset(torch.tensor([[o[0]] for o in train]), torch.tensor([[o[1]] for o in train], dtype=torch.float32)), batch_size=BATCH_SIZE, shuffle=True, num_workers=N_WORKERS)
	x_to_c_test_loader = DataLoader(TensorDataset(torch.tensor([[o[0]] for o in test]), torch.tensor([[o[1]] for o in test], dtype=torch.float32)), batch_size=BATCH_SIZE, shuffle=False, num_workers=N_WORKERS)

	x_to_c_model = Model(layer_sizes=[24, 128, 128, n_concepts], 
				  activation=nn.ReLU,
				  final_activation=None,
				  lr=hparams[0],
				  loss=nn.functional.binary_cross_entropy_with_logits, device='cuda')

	train_loss, test_loss = x_to_c_model.fit(x_to_c_train_loader, None, n_epoch=hparams[1])

	concept_labels = [obs_to_concepts(o) for o in test]
	concept_logits = x_to_c_model.predict(x_to_c_test_loader).cpu()
	concept_preds = torch.sigmoid(concept_logits) > 0.5

	print(f'Concept 0 acc: {accuracy_score([c[0] for c in concept_labels], concept_preds[:,0].cpu())}')
	print(f'Concept 1 acc: {accuracy_score([c[1] for c in concept_labels], concept_preds[:,1].cpu())}')
	print(f'Concept 2 acc: {accuracy_score([c[2] for c in concept_labels], concept_preds[:,2].cpu())}')
	#pd.DataFrame({'train': train_loss, 'test': test_loss}).plot(
	#	title=f'Features to Concepts - {n_concepts}, {seed}')
	#plt.show()
	
	return x_to_c_model

def cpred_to_y(train, test, x_to_c_model, seed, hparams):
	
	n_concepts = len(obs_to_concepts(train[0]))
	x_to_c_train_loader = DataLoader(TensorDataset(torch.tensor([[o[0]] for o in train]), torch.tensor([[o[1]] for o in train], dtype=torch.float32)), batch_size=BATCH_SIZE, shuffle=True, num_workers=N_WORKERS)
	x_to_c_test_loader = DataLoader(TensorDataset(torch.tensor([[o[0]] for o in test]), torch.tensor([[o[1]] for o in test], dtype=torch.float32)), batch_size=BATCH_SIZE, shuffle=False, num_workers=N_WORKERS)

	cpreds_train = torch.sigmoid(x_to_c_model.predict(x_to_c_train_loader))
	cpreds_test = torch.sigmoid(x_to_c_model.predict(x_to_c_test_loader))

	#cpreds_train = x_to_c_model.predict(x_to_c_train_loader)
	#cpreds_test = x_to_c_model.predict(x_to_c_test_loader)

	cpred_to_y_train_loader = DataLoader(
		TensorDataset(cpreds_train, 
					  torch.tensor([[o[2]] for o in train], dtype=torch.float32)),
					  batch_size=BATCH_SIZE, shuffle=True)
	cpred_to_y_test_loader = DataLoader(
		TensorDataset(cpreds_test, 
					  torch.tensor([[o[2]] for o in test], dtype=torch.float32)),
					  batch_size=BATCH_SIZE, shuffle=False)

	cpred_to_y_model = Model(layer_sizes=[n_concepts, 128, 128, 1], 
							 activation=nn.ReLU,
							 final_activation=None,
							 lr=hparams[0], 
							 loss=nn.functional.binary_cross_entropy_with_logits)

	train_loss, test_loss = cpred_to_y_model.fit(
		cpred_to_y_train_loader, cpred_to_y_test_loader, n_epoch=hparams[1])

	#pd.DataFrame({'train': train_loss, 'test': test_loss}).plot(
	#    title=f'Concepts Preds to Task - {n_concepts}, {seed}')
	#plt.show()
	
	pred_logits_test = cpred_to_y_model.predict(cpred_to_y_test_loader).cpu()
	preds_test = torch.sigmoid(pred_logits_test) > 0.5
	acc_test = accuracy_score([o[2] for o in test], preds_test)
	print(f'Test: {acc_test}')

	pred_logits_train = cpred_to_y_model.predict(cpreds_train).cpu()
	preds_train = torch.sigmoid(pred_logits_train) > 0.5
	acc_train = accuracy_score([o[2] for o in train], preds_train)
	print(f'Train: {acc_train}')
	return acc_train, acc_test

dataa = generate_dataset(seedyseed)
dataset = toydataset(generate_dataset(seedyseed))
train, test = dataset.get_splits()
HPARAMS = ((0.001, 10), 
		   (0.001, 200))

x_to_c_model = x_to_c(train, test, 1111, HPARAMS[0])
cpred_to_y(train, test, x_to_c_model, 1111, HPARAMS[1])