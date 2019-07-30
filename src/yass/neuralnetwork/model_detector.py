import numpy as np
import os 
import torch
from torch import nn, optim
import torch.utils.data as Data
from torch.nn import functional as F
from torch import distributions


class Detect(nn.Module):
    def __init__(self, n_filters, spike_size, channel_index):
        super(Detect, self).__init__()
        
        self.spike_size = spike_size
        self.channel_index = channel_index
        n_neigh = self.channel_index.shape[1]
        
        feat1, feat2, feat3 = n_filters

        self.temporal_filter1 = nn.Sequential(         # input shape (1, 28, 28)
            nn.Conv2d(
                in_channels=1,              # input height
                out_channels=feat1,            # n_filters
                kernel_size=[spike_size, 1],              # filter size
                stride=1,                   # filter movement/step
                padding=[(self.spike_size-1)//2, 0],                # if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1
            ),                              # output shape (16, 28, 28)
            nn.ReLU(),                      # activation
        )

        self.temporal_filter2 = nn.Sequential(         # input shape (16, 14, 14)
            nn.Conv2d(feat1, feat2, [1, 1], 1, 0),     # output shape (32, 14, 14)
            nn.ReLU(),                      # activation
        )
        
        #self.spatial_filter = nn.Sequential(         # input shape (16, 14, 14)
        #    nn.Conv1d(feat2, feat3, [1, n_neigh], 1, 0),     # output shape (32, 14, 14)
        #    nn.ReLU(),                      # activation
        #)
        self.out = nn.Linear(feat3*n_neigh, 1)

    def forward(self, x):

        x = x[:, None]
        x = self.temporal_filter1(x)
        x = self.temporal_filter2(x)[:, :, 0]
        x = x.reshape(x.shape[0], -1)
        x = self.out(x)
        output = torch.sigmoid(x)

        return output, x   # return x for visualization
    
    def forward_recording(self, recording_tensor):

        x = recording_tensor[None, None]
        x = self.temporal_filter1(x)
        x = self.temporal_filter2(x)

        zero_buff = torch.zeros(
            [1, x.shape[1], x.shape[2], 1]).to(x.device)
        x = torch.cat((x, zero_buff), 3)[0]
        x = x[:, :, self.channel_index].permute(1, 2, 0, 3)
        x = self.out(x.reshape(
            recording_tensor.shape[0]*recording_tensor.shape[1], -1))
        x = x.reshape(recording_tensor.shape[0],
                      recording_tensor.shape[1])
        
        return x

    def get_spike_times(self, recording_tensor, max_window=5, threshold=0.5, buffer=None):
        
        probs = self.forward_recording(recording_tensor)
        
        maxpool = torch.nn.MaxPool2d(kernel_size=[max_window, 1], stride=1, padding=[(max_window-1)//2, 0])
        temporal_max = maxpool(probs[None])[0] - 1e-8

        spike_index_torch = torch.nonzero(
            (probs >= temporal_max) & (probs > np.log(threshold / (1 - threshold))))
        
        # remove edge spikes
        if buffer is None:
            buffer = self.spike_size//2

        spike_index_torch = spike_index_torch[
            (spike_index_torch[:, 0] > buffer) & 
            (spike_index_torch[:, 0] < recording_tensor.shape[0] - buffer)]

        wf_t_range = torch.arange(
            -(self.spike_size//2), self.spike_size//2+1).to(spike_index_torch.device)
        time_index = spike_index_torch[:, 0][:, None] + wf_t_range
        channel_index = spike_index_torch[:, 1][:,None].repeat((1, self.spike_size))
        wf = recording_tensor[time_index, channel_index]

        return spike_index_torch, wf
    
    def train(self, fname_save, DetectTD, n_train=50000, n_test=1000, EPOCH=1000, BATCH_SIZE=512, LR=0.0001):

        print('Training NN detector')

        if os.path.exists(fname_save):
            return

        self.temporal_filter1[0].padding = [0, 0]
            
        optimizer = torch.optim.Adam(self.parameters(), lr=LR)   # optimize all cnn parameters
        loss_func = nn.BCELoss()                       # the target label is not one-hotted
        
        x_train, y_train, y_test_clean = DetectTD.make_training_data(n_train)
        train = Data.TensorDataset(torch.FloatTensor(x_train), torch.FloatTensor(y_train))
        train_loader = Data.DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)
        
        x_test, y_test, x_test_clean = DetectTD.make_training_data(n_test)
        
        # training and testing
        for epoch in range(EPOCH):
            for step, (b_x, b_y) in enumerate(train_loader):   # gives batch data, normalize x when iterate train_loader
                est = self(b_x.cuda())[0]
                loss = loss_func(est[:, 0], b_y.cuda())   # cross entropy loss
                optimizer.zero_grad()           # clear gradients for this training step
                loss.backward()                 # backpropagation, compute gradients
                optimizer.step()                # apply gradients

                if step % 100 == 0:
                    est_test = self(torch.FloatTensor(x_test).cuda())[0].cpu().data.numpy()
                    pred = est_test[:, 0] > 0.5
                    correct = (pred == y_test)
                    tp = np.mean(correct[y_test==1])
                    fp = 1-np.mean(correct[y_test==0])
                    l2_loss = np.mean((pred == y_test))
                    print('Epoch: ', epoch,
                          '| train loss: %.4f' % loss.cpu().data.numpy(),
                          '| TP : %.4f' % tp,
                          '| FP : %.4f' % fp)

        # save model
        torch.save(self.state_dict(), fname_save)
                
    def load(self, fname_model):
        self.load_state_dict(torch.load(fname_model))


class Detect_Single(nn.Module):
    def __init__(self, spike_size):
        super(Detect_Single, self).__init__()
        
        self.spike_size = spike_size
        
        feat1 = 16
        feat2 = 8
        feat3 = 8

        self.temporal_filter1 = nn.Sequential(         # input shape (1, 28, 28)
            nn.Conv1d(
                in_channels=1,              # input height
                out_channels=feat1,            # n_filters
                kernel_size=spike_size,              # filter size
                stride=1,                   # filter movement/step
                padding=(self.spike_size-1)//2,                # if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1
            ),                              # output shape (16, 28, 28)
            nn.ReLU(),                      # activation
        )
        self.temporal_filter2 = nn.Sequential(         # input shape (16, 14, 14)
            nn.Conv1d(feat1, feat2, 1, 1, 0),     # output shape (32, 14, 14)
            nn.ReLU(),                      # activation
        )
        self.out = nn.Linear(feat3, 1)

    def forward(self, x):
        x = x[:, None]
        x = self.temporal_filter1(x)
        x = self.temporal_filter2(x)
        x = self.out(x[:, :, 0])
        output = torch.sigmoid(x)

        return output, x   # return x for visualization
    
    def forward_recording(self, recording):
        
        x = recording.transpose(1,0)
        x = self.temporal_filter1(x[:, None])
        x = self.temporal_filter2(x)
        x = self.out(x.permute(2,0,1).reshape(-1, x.shape[1])).reshape(recording.shape[0], recording.shape[1])

        return x

    def get_spike_times(self, recording, max_window=5, threshold=0.5, buffer=None):
        
        recording_torch = torch.FloatTensor(recording)
        x = self.forward_recording(recording_torch).permute(1,0)[:, None]

        maxpool = torch.nn.MaxPool1d(kernel_size=max_window, stride=1, padding=(max_window-1)//2)
        temporal_max = (maxpool(x)[:, 0] - 1e-8).transpose(0,1)
        x = x[:, 0].transpose(0,1)

        spike_index_torch = torch.nonzero((x >= temporal_max) & (x > np.log(threshold / (1 - threshold))))

        # remove edge spikes
        if buffer is None:
            buffer = self.spike_size//2

        spike_index_torch = spike_index_torch[
            (spike_index_torch[:, 0] > buffer) & 
            (spike_index_torch[:, 0] < recording_torch.shape[0] - buffer)]

        time_index = (spike_index_torch[:, 0][:, None] + 
                      torch.arange(-(self.spike_size//2), self.spike_size//2+1))
        channel_index = spike_index_torch[:, 1][:,None].repeat((1, self.spike_size))
        wf = recording_torch[time_index, channel_index]

        return spike_index_torch, wf
    
    
    def train(self, fname_save, DetectTD, n_train=50000, n_test=1000, EPOCH=1000, BATCH_SIZE=512, LR=0.0001):

        self.temporal_filter1[0].padding = (0,)

        optimizer = torch.optim.Adam(self.parameters(), lr=LR)   # optimize all cnn parameters
        loss_func = nn.BCELoss()                       # the target label is not one-hotted
        
        x_train, y_train, y_test_clean = DetectTD.make_training_data(n_train)
        train = Data.TensorDataset(torch.FloatTensor(x_train), torch.FloatTensor(y_train))
        train_loader = Data.DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)

        x_test, y_test, x_test_clean = DetectTD.make_training_data(n_test)

        # training and testing
        for epoch in range(EPOCH):
            for step, (b_x, b_y) in enumerate(train_loader):   # gives batch data, normalize x when iterate train_loader
                est = self(b_x[:,:,0].cuda())[0]
                loss = loss_func(est[:, 0], b_y.cuda())   # cross entropy loss
                optimizer.zero_grad()           # clear gradients for this training step
                loss.backward()                 # backpropagation, compute gradients
                optimizer.step()                # apply gradients

                if step % 100 == 0:
                    est_test = self(torch.FloatTensor(x_test[:,:,0]).cuda())[0].cpu().data.numpy()
                    pred = est_test[:, 0] > 0.5
                    correct = (pred == y_test)
                    tp = np.mean(correct[y_test==1])
                    fp = 1-np.mean(correct[y_test==0])
                    l2_loss = np.mean((pred == y_test))
                    print('Epoch: ', epoch,
                          '| train loss: %.4f' % loss.cpu().data.numpy(),
                          '| TP : %.4f' % tp,
                          '| FP : %.4f' % fp)
        
        # save model
        torch.save(self.state_dict(), fname_save)
                
    def load(self, fname_model):
        self.load_state_dict(torch.load(fname_model))
