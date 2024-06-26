import ray
import json
import torch, argparse, os, time, sys, shutil, logging
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from torchsummary import summary
from tqdm import tqdm
from skimage.transform import resize
from numpy.fft import fftn, fftshift

import torch.nn.functional as F
import torch.nn as nn

from torch.nn import MaxPool2d, ReLU, Linear, Upsample

from matplotlib import pyplot as plt
import matplotlib as mpl

import biotorch
from math import log

from deephyper.problem import HpProblem
from deephyper.search.hps import CBO
from deephyper.evaluator import Evaluator
from deephyper.evaluator.callback import TqdmCallback

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

import argparse

numIterations = 100

#adding a parseable parameter for training algorithm. 
parser = argparse.ArgumentParser(description='PtychoNN.')
parser.add_argument('-train_alg',type=str, default="BP")

args, unparsed = parser.parse_known_args()

# learning algorithms supported in this experiment. 
if args.train_alg=="BP":
    from torch.nn import Conv2d
elif args.train_alg=="fa":
    from biotorch.layers.fa import Conv2d
elif args.train_alg=="dfa":
    from biotorch.layers.dfa import Conv2d
elif args.train_alg=="usf":
    from biotorch.layers.usf import Conv2d
elif args.train_alg=="frsf":
    from biotorch.layers.frsf import Conv2d
elif args.train_alg=="brsf":
    from biotorch.layers.brsf import Conv2d
else :
    print(train_alg, 'is not supported')

#epochs = 60

batch = 64


h,w = 64,64
nlines = 100 #How many lines of data to use for training?
nltest = 60 #How many lines for the test set?

n_valid = 805 #How much to reserve for validation

path = os.getcwd()

#nconv = 16

def insertConv(layers, depth, in_channels, out_channels):
    layers.append(Conv2d(in_channels, out_channels, 3, stride=1, padding=(1,1)))
    layers.append(torch.nn.ReLU())
    for i in range(depth-1):
        layers.append(Conv2d(out_channels, out_channels, 3, stride=1, padding=(1,1)))
        layers.append(torch.nn.ReLU())
    
    return layers 

class PtychoNN(torch.nn.Module):  
    def __init__(self, encode_depth=2, decode_depth_1=2, decode_depth_2=2, en_filters=[], d1_filters=[], d2_filters=[]): 
        super().__init__()

        #define layers in a list:
        elayers = []
        elayers.append(Conv2d(in_channels=1, out_channels=en_filters[0], kernel_size=3, stride=1, padding=(1,1)))
        elayers.append(ReLU())
        elayers = insertConv(elayers, encode_depth-1, en_filters[0], en_filters[0])
        elayers.append(MaxPool2d((2,2)))
        elayers = insertConv(elayers, encode_depth, en_filters[0], en_filters[1])
        elayers.append(MaxPool2d((2,2)))
        elayers = insertConv(elayers, encode_depth, en_filters[1], en_filters[2])
        elayers.append(MaxPool2d((2,2)))

        self.encoder = torch.nn.Sequential(*elayers)
        
        
        d1layers = []
        d1layers = insertConv(d1layers,decode_depth_1, en_filters[2], d1_filters[0])
        d1layers.append(Upsample(scale_factor=2, mode='bilinear'))
        d1layers = insertConv(d1layers, decode_depth_1, d1_filters[0], d1_filters[1])
        d1layers.append(Upsample(scale_factor=2, mode='bilinear'))
        d1layers = insertConv(d1layers, decode_depth_1, d1_filters[1], d1_filters[2])
        d1layers.append(Upsample(scale_factor=2, mode='bilinear'))
        d1layers.append(Conv2d(d1_filters[2], 1, 3, stride=1, padding=(1,1)))
        d1layers.append(torch.nn.Sigmoid())

        self.decoder1 = torch.nn.Sequential(*d1layers)
       

        
        d2layers = []
        d2layers = insertConv(d2layers, decode_depth_2, en_filters[2], d2_filters[0])
        d2layers.append(Upsample(scale_factor=2, mode='bilinear'))
        d2layers = insertConv(d2layers, decode_depth_2, d2_filters[0], d2_filters[1])
        d2layers.append(Upsample(scale_factor=2, mode='bilinear'))
        d2layers = insertConv(d2layers, decode_depth_2, d2_filters[1], d2_filters[2])
        d2layers.append(Upsample(scale_factor=2, mode='bilinear'))
        d2layers.append(Conv2d(d2_filters[2], 1, 3, stride=1, padding=(1,1)))
        d2layers.append(torch.nn.Tanh())

        self.decoder2 = torch.nn.Sequential(*d2layers)


    def forward(self,x):
        x1 = self.encoder(x)
        amp = self.decoder1(x1)
        ph = self.decoder2(x1)

        #Restore -pi to pi range
        ph = ph*np.pi #Using tanh activation (-1 to 1) for phase so multiply by pi

        return amp,ph


#load the data for braggNN
def load_data():

    data_diffr = np.load('data/20191008_39_diff.npz')['arr_0']
    real_space = np.load('data/20191008_39_amp_pha_10nm_full.npy')#, allow_pickle=True)
    amp = np.abs(real_space)
    ph = np.angle(real_space)
    amp.shape


    data_diffr = np.load('data/20191008_39_diff.npz')['arr_0']
    real_space = np.load('data/20191008_39_amp_pha_10nm_full.npy')#, allow_pickle=True)
    amp = np.abs(real_space)
    ph = np.angle(real_space)
    amp.shape

    #plt.matshow(np.log10(data_diffr[0,0]))

    data_diffr_red = np.zeros((data_diffr.shape[0],data_diffr.shape[1],64,64), float)
    for i in range(data_diffr.shape[0]):
        for j in range(data_diffr.shape[1]):
            data_diffr_red[i,j] = resize(data_diffr[i,j,32:-32,32:-32],(64,64),preserve_range=True, anti_aliasing=True)
            data_diffr_red[i,j] = np.where(data_diffr_red[i,j]<3,0,data_diffr_red[i,j])

    tst_strt = amp.shape[0]-nltest #Where to index from

    X_train = data_diffr_red[:nlines,:].reshape(-1,h,w)[:,np.newaxis,:,:]
    X_test = data_diffr_red[tst_strt:,tst_strt:].reshape(-1,h,w)[:,np.newaxis,:,:]
    Y_I_train = amp[:nlines,:].reshape(-1,h,w)[:,np.newaxis,:,:]
    Y_I_test = amp[tst_strt:,tst_strt:].reshape(-1,h,w)[:,np.newaxis,:,:]
    Y_phi_train = ph[:nlines,:].reshape(-1,h,w)[:,np.newaxis,:,:]
    Y_phi_test = ph[tst_strt:,tst_strt:].reshape(-1,h,w)[:,np.newaxis,:,:]

    ntrain = X_train.shape[0]*X_train.shape[1]
    ntest = X_test.shape[0]*X_test.shape[1]



    X_train, Y_I_train, Y_phi_train = shuffle(X_train, Y_I_train, Y_phi_train, random_state=0)

    #Training data
    X_train_tensor = torch.Tensor(X_train)
    Y_I_train_tensor = torch.Tensor(Y_I_train)
    Y_phi_train_tensor = torch.Tensor(Y_phi_train)

    #Test data
    X_test_tensor = torch.Tensor(X_test)
    Y_I_test_tensor = torch.Tensor(Y_I_test)
    Y_phi_test_tensor = torch.Tensor(Y_phi_test)



    train_data = TensorDataset(X_train_tensor,Y_I_train_tensor,Y_phi_train_tensor)
    test_data = TensorDataset(X_test_tensor)

    n_train = X_train_tensor.shape[0]
    train_data2, valid_data = torch.utils.data.random_split(train_data,[n_train-n_valid,n_valid])

    #download and load training data
    trainloader = DataLoader(train_data2, batch_size=batch, shuffle=True, num_workers=4)
    validloader = DataLoader(valid_data, batch_size=batch, shuffle=True, num_workers=4)
    testloader = DataLoader(test_data, batch_size=batch, shuffle=False, num_workers=4)

    iterations_per_epoch = np.floor((n_train-n_valid)/batch)+1 #Final batch will be less than batch size
    step_size = 6*iterations_per_epoch
    return trainloader, validloader, testloader, step_size

# Define the run function for the DeepHyper optimizer. 
def run(config: dict):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loss_func = torch.nn.L1Loss()
    
    metrics = {'losses':[],'val_losses':[], 'lrs':[], 'best_val_loss' : np.inf, 'best_amp_loss': np.inf, 'best_phase_loss': np.inf}

    en_filters = []
    d1_filters = []
    d2_filters = []

    en_filters.append(config["en1_filter"])
    en_filters.append(config["en2_filter"])
    en_filters.append(config["en3_filter"])
    
    d1_filters.append(config["de11_filter"])
    d1_filters.append(config["de12_filter"])
    d1_filters.append(config["de13_filter"])
    
    d2_filters.append(config["de21_filter"])
    d2_filters.append(config["de22_filter"])
    d2_filters.append(config["de23_filter"])

    edepth = config["edepth"]
    d1depth = config["d1depth"]
    d2depth = config["d2depth"]
    lrate = config["learning_rate"]

    epochs = config["epochs"]

    embed_dim = 64
    model = PtychoNN(encode_depth=edepth, decode_depth_1=d1depth, decode_depth_2=d2depth, en_filters=en_filters, d1_filters=d1_filters, d2_filters=d2_filters)

    #summary(model,(1,64,64),device="cpu")
    
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model) #Default all devices

    model = model.to(device)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer = torch.optim.Adam(model.parameters(), lr=lrate) 

    # load data from the DataLoader 
    dl_train, dl_valid, dl_test, step_size = load_data()
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=lrate/10, max_lr=lrate, step_size_up=step_size, cycle_momentum=False, mode='triangular2')

    #train
    for epoch in range(epochs):
        #iterate through all the data in the training dataset
        tot_loss = 0.0
        amp_loss = 0.0
        phs_loss = 0.0
        for i, (tr_data, tr_amp, tr_phs) in enumerate(dl_train):
            
            #forward pass
            pred_amp, pred_phs = model.forward(tr_data.to(device))

            #compute the individual loss for each of the functions: 
            loss_amp = loss_func(pred_amp, tr_amp.to(device))
            loss_phs = loss_func(pred_phs, tr_phs.to(device))

            #compute the total loss: 
            loss_total = loss_amp + loss_phs

            #backprop
            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()
        
            tot_loss += loss_total.detach().item()
            amp_loss += loss_amp.detach().item()
            phs_loss += loss_phs.detach().item()

            #Update the LR according to the schedule -- CyclicLR updates each batch
            scheduler.step()

        metrics['losses'].append([tot_loss/i,amp_loss/i,phs_loss/i])

        # validate
        tot_val_loss = 0.0
        val_loss_amp = 0.0
        val_loss_ph = 0.0
        
        for j, (ft_images,amps,phs) in enumerate(dl_valid):
        
            ft_images = ft_images.to(device)
            amps = amps.to(device)
            phs = phs.to(device)
            pred_amps, pred_phs = model(ft_images) #Forward pass

            val_loss_a = loss_func(pred_amps,amps)
            val_loss_p = loss_func(pred_phs,phs)
            val_loss = val_loss_a + val_loss_p

            tot_val_loss += val_loss.detach().item()
            val_loss_amp += val_loss_a.detach().item()
            val_loss_ph += val_loss_p.detach().item()
        
        metrics['val_losses'].append([tot_val_loss/j,val_loss_amp/j,val_loss_ph/j])

        #Update the metrics for the individual phase and amplitude
        if(val_loss_amp/j < metrics['best_amp_loss']):
            metrics['best_amp_loss'] = val_loss_amp/j

        #Update the metrics for the individual phase and amplitude
        if(val_loss_ph/j < metrics['best_phase_loss']):
            metrics['best_phase_loss'] = val_loss_ph/j

        #Update saved model if val loss is lower
        if(tot_val_loss/j<metrics['best_val_loss']):
            #print("Saving improved model after Val Loss improved from %.5f to %.5f" %(metrics['best_val_loss'],tot_val_loss/j))
            metrics['best_val_loss'] = tot_val_loss/j
    
    print("Phase: %.2f, Amplitude: %.2f, Total: %.2f" %( float(metrics['best_phase_loss']), float(metrics['best_amp_loss']), float(metrics['best_phase_loss']+metrics['best_amp_loss'])))
    return -metrics['best_val_loss']


# define the evaluator function for Deephyper
def get_evaluator(run_function):
    # Default arguments for Ray: 1 worker and 1 worker per evaluation
    method_kwargs = {
        "num_cpus": 1,
        "num_cpus_per_task": 1,
        "callbacks": [TqdmCallback()]
    }

    # If GPU devices are detected then it will create 'n_gpus' workers
    # and use 1 worker for each evaluation
    if is_gpu_available:
        method_kwargs["num_cpus"] = n_gpus
        method_kwargs["num_gpus"] = n_gpus
        method_kwargs["num_cpus_per_task"] = 1
        method_kwargs["num_gpus_per_task"] = 1

    evaluator = Evaluator.create(
        run_function,
        method="ray",
        method_kwargs=method_kwargs
    )
    print(f"Created new evaluator with {evaluator.num_workers} worker{'s' if evaluator.num_workers > 1 else ''} and config: {method_kwargs}", )

    return evaluator

is_gpu_available = torch.cuda.is_available()
n_gpus = torch.cuda.device_count()
#n_gpus = 1

#load data
# hyperparameters of interest
#1. number of neurons per layer
#2. number of layers
#3. learning rules
#4. temp - slope of the discretization function

problem = HpProblem()
#epoch
problem.add_hyperparameter((1, 100), "epochs", default_value=60)
#learning rate
problem.add_hyperparameter((0.0001, 0.1, "log-uniform"), "learning_rate", default_value=0.005)
#encoder filter-1
problem.add_hyperparameter((1, 256), "en1_filter", default_value=16)
#encoder filter-2
problem.add_hyperparameter((1, 256), "en2_filter", default_value=16)
#encoder filter-3
problem.add_hyperparameter((1, 256), "en3_filter", default_value=16)
#decoder-1 filter-1
problem.add_hyperparameter((1, 256), "de11_filter", default_value=16)
#decoder-1 filter-2
problem.add_hyperparameter((1, 256), "de12_filter", default_value=16)
#decoder-1 filter-3
problem.add_hyperparameter((1, 256), "de13_filter", default_value=16)
#decoder-2 filter-1
problem.add_hyperparameter((1, 256), "de21_filter", default_value=16)
#decoder-2 filter-2
problem.add_hyperparameter((1, 256), "de22_filter", default_value=16)
#decoder-2 filter-3
problem.add_hyperparameter((1, 256), "de23_filter", default_value=16)
#encoder depth
problem.add_hyperparameter((1,10), "edepth", default_value=2)
#decoder-1 depth
problem.add_hyperparameter((1,10), "d1depth", default_value=2)
#decoder-2 depth
problem.add_hyperparameter((1,10), "d2depth", default_value=2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#create the evaluator
evaluator_1 = get_evaluator(run)

#trigger the search
search = CBO(problem, evaluator_1, log_dir='./Results/')

# load a surrogate
search.fit_surrogate("./Results/results1.csv")

#number of evals
results = search.search(max_evals=numIterations)
