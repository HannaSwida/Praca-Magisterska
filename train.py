import sys
import time
import torch
from tqdm import tqdm
import numpy as np
import GPUtil
from termcolor import colored
import os
from model import Model

from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn

from voxceleb_loader_02 import VCset

def train_one_epoch(data_loader, net, device, criterion, optimizer, scaler):
    
    #NETS, OPTIMIZERS

    n_samples = 0
    errs = 0
    running_loss = 0

    data_loader = tqdm(data_loader)

    net.train(True)

    m = nn.Sigmoid()

    for i, data in enumerate(data_loader):
  
        # get the inputs; data is a list of [inputs, labels]
        inputs = data
        n = inputs[0].size(0)

        labels0 = torch.zeros(n,1)
        labels1 = torch.ones(n,1)
        labels = torch.cat((labels0, labels1))

        
        np_labels = labels.detach().numpy()
        #
        for i in range(len(inputs)):
            inputs[i] = inputs[i].to(device)

        labels = labels.to(device)

        #outputs, eA, eB = net(inputs.float(), lenghts, dropout)

        with torch.cuda.amp.autocast():
            outputs = net(inputs[0].float(), inputs[1].float(), inputs[2].float()) 
            outputs = torch.cat((outputs[0], outputs[1]))
            outputs2 = m(outputs)

            loss = criterion(torch.squeeze(outputs), torch.squeeze(labels))
  
        n_samples += outputs.size(0)
        np_outputs = outputs2.cpu().detach().numpy()

        e = np.abs(np_labels - np_outputs)>0.1
        errs += np.sum(e)            
                
        optimizer.zero_grad() # zero the parameter gradients
        loss.backward()

        optimizer.step()

        # print statistics
        running_loss += loss.item()

    net.train(False)  # Set model to evaluate mode
        
    return errs, running_loss/i

def get_device():
    
    n_gpu = 0
    GPUtil.showUtilization()
    deviceIDs = GPUtil.getAvailable(order='load', limit=8, maxLoad=0.9, maxMemory=0.7)


    if deviceIDs == []:
        print(colored("CPU selected.","red"))
        device = torch.device('cpu')
    else:
        print(colored("GPU" + str(deviceIDs[0]) + " selected.", "green"))
        device = torch.device('cuda:'+str(deviceIDs[0]))
        n_gpu = len(deviceIDs)

    return device, n_gpu, deviceIDs

def weights_init(m):
    if isinstance(m, nn.Conv1d):
        nn.init.xavier_normal_(m.weight.data)

def load_model(model_path):
    model = Model()  # Create an instance of your model (assuming Model class exists)
    checkpoint = torch.load(model_path)  # Load the checkpoint
    model.load_state_dict(checkpoint['model_state_dict'])  # Load the model state dict

    # Additional code for any other components you want to load from the checkpoint
    # For example, optimizer, epoch, etc.

    model.eval()  # Set the model to evaluation mode

    return model

def main(argv):
    batch_size = 128
    n_batches = 1000
    n_workers = 1
    n_epochs = 1500
    lr = 0.001

    input_dirs_file = '../../source_vox1dir.txt'
    outputdir = '../checkpoints/MiModel/02s/vox1_1000B_02s'

    device, n_gpu, deviceIDs = get_device()

    if n_gpu == 0:
        sys.exit(2)

    tr_data = VCset(input_dirs_file, n_batches=n_batches, chunk_size=0.4)
    tr_loader = DataLoader(tr_data, batch_size=batch_size, pin_memory=True, num_workers=n_workers)

    net = Model()
    net = net.to(device=device)

    net.apply(weights_init)

    if n_gpu > 1:
        net = nn.DataParallel(net, device_ids=deviceIDs)

    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, nesterov=True) #, weight_decay=0.01)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.99)
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(n_epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        errs = 0

        start = time.time()

        errs, running_loss = train_one_epoch(tr_loader, net, device, criterion, optimizer, scaler)
            
        scheduler.step()
        
        end = time.time()

        print('Time elapsed: ' + "{:.2f}".format(end - start) + "s")

        pad = ' '
        print('[%d] %s loss: %.3f     err: %.3f' %(epoch, 20*pad, running_loss, float(errs)))
        

        if outputdir != "":
            path = outputdir + "/" + str(epoch) + ".tar"

            if n_gpu < 2:
                model_state_dict = net.state_dict()
            else:
                model_state_dict = net.module.state_dict()

            torch.save({
            'epoch': epoch,
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': optimizer.state_dict(),
            #'loss': loss,
            'nlabels' : tr_data.get_nlabels()
            }, path)  


        # Save the encoder model
        if outputdir != "":
            encoder_save_path = outputdir + "/encoder_" + str(epoch) + ".pth"

            if n_gpu < 2:
                encoder_state_dict = net.encoder.state_dict()
            else:
                encoder_state_dict = net.module.encoder.state_dict()

            torch.save(encoder_state_dict, encoder_save_path)
            print("Encoder model saved at:", encoder_save_path)

        print('\nFinished Training')


if __name__ == "__main__":

   main(sys.argv[1:])
