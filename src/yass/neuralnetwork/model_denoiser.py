import numpy as np
import os
import torch
from torch import nn, optim
import torch.utils.data as Data
from torch.nn import functional as F
from torch import distributions

class Denoise(nn.Module):
    def __init__(self, n_filters, filter_sizes, spike_size):
        super(Denoise, self).__init__()
        
        feat1, feat2, feat3 = n_filters
        size1, size2, size3 = filter_sizes
        
        self.conv1 = nn.Sequential(         # input shape (1, 28, 28)
            nn.Conv1d(
                in_channels=1,              # input height
                out_channels=feat1,            # n_filters
                kernel_size=size1,              # filter size
                stride=1,                   # filter movement/step
                padding=0,                # if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1
            ),                              # output shape (16, 28, 28)
            nn.ReLU(),                      # activation
        ).cuda()

        self.conv2 = nn.Sequential(         # input shape (16, 14, 14)
            nn.Conv1d(feat1, feat2, size2, 1, 0),     # output shape (32, 14, 14)
            nn.ReLU(),                      # activation
        ).cuda()

        self.conv3 = nn.Sequential(         # input shape (16, 14, 14)
            nn.Conv1d(feat2, feat3, size3, 1, 0),     # output shape (32, 14, 14)
            nn.ReLU(),                      # activation
        ).cuda()

        #n_input_feat = feat3*(61-size1-size2-size3+3)
        n_input_feat = feat2*(spike_size-size1-size2+2)
        self.out = nn.Linear(n_input_feat, spike_size).cuda()

    def forward(self, x):
        x = x[:, None]
        x = self.conv1(x)
        x = self.conv2(x)
        #x = self.conv3(x)
        x = x.view(x.shape[0], -1)
        output = self.out(x)

        return output, x   # return x for visualization

    def train(self, fname_save, DenoTD, n_train=50000, n_test=500, EPOCH=500, BATCH_SIZE=512, LR=0.0001):

        print('Training NN denoiser')

        if os.path.exists(fname_save):
            return

        optimizer = torch.optim.Adam(self.parameters(), lr=LR)   # optimize all cnn parameters
        loss_func = nn.MSELoss()                       # the target label is not one-hotted

        wf_col_train, wf_clean_train = DenoTD.make_training_data(n_train)
        train = Data.TensorDataset(torch.FloatTensor(wf_col_train), torch.FloatTensor(wf_clean_train))
        train_loader = Data.DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)

        wf_col_test, wf_clean_test = DenoTD.make_training_data(n_test)

        # training and testing
        for epoch in range(EPOCH):
            for step, (b_x, b_y) in enumerate(train_loader):   # gives batch data, normalize x when iterate train_loader
                est = self(b_x.cuda())[0]
                loss = loss_func(est, b_y.cuda())   # cross entropy loss
                optimizer.zero_grad()           # clear gradients for this training step
                loss.backward()                 # backpropagation, compute gradients
                optimizer.step()                # apply gradients

                if step % 100 == 0:
                    est_test = self(torch.FloatTensor(wf_col_test).cuda())[0]
                    l2_loss = np.mean(np.square(est_test.cpu().data.numpy() - wf_clean_test))
                    print('Epoch: ', epoch, '| train loss: %.4f' % loss.cpu().data.numpy(), '| test accuracy: %.4f' % l2_loss)
    
        # save model
        torch.save(self.state_dict(), fname_save)
                
    def load(self, fname_model):
        self.load_state_dict(torch.load(fname_model))
