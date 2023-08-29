import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
#from dnn_models import SincNet, MLP, MLP_mode

class SincConv(nn.Module):
    """Sinc-based convolution
    Parameters
    ----------
    in_channels : `int`
        Number of input channels. Must be 1.
    out_channels : `int`
        Number of filters.
    kernel_size : `int`
        Filter length.
    sample_rate : `int`, optional
        Sample rate. Defaults to 16000.
    Usage
    -----
    See `torch.nn.Conv1d`
    Reference
    ---------
    Mirco Ravanelli, Yoshua Bengio,
    "Speaker Recognition from raw waveform with SincNet".
    https://arxiv.org/abs/1808.00158
    """

    @staticmethod
    def to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    @staticmethod
    def to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)

    def __init__(self, out_channels, kernel_size, sample_rate=16000, 
                 stride=1, padding=0, dilation=1, min_low_hz=50, min_band_hz=50):

        super(SincConv,self).__init__()

        
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        
        # Forcing the filters to be odd (i.e, perfectly symmetrics)
        if kernel_size%2==0:
            self.kernel_size=self.kernel_size+1
            
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        
        self.sample_rate = sample_rate
        self.min_low_hz = min_low_hz
        self.min_band_hz = min_band_hz

        # initialize filterbanks such that they are equally spaced in Mel scale
        low_hz = 30
        high_hz = self.sample_rate / 2 - (self.min_low_hz + self.min_band_hz)

        mel = np.linspace(self.to_mel(low_hz),
                          self.to_mel(high_hz),
                          self.out_channels + 1)
        hz = self.to_hz(mel)
        

        # filter lower frequency (out_channels, 1)
        self.low_hz_ = nn.Parameter(torch.Tensor(hz[:-1]).view(-1, 1))

        # filter frequency band (out_channels, 1)
        self.band_hz_ = nn.Parameter(torch.Tensor(np.diff(hz)).view(-1, 1))

        # Hamming window
        #self.window_ = torch.hamming_window(self.kernel_size)
        n_lin=torch.linspace(0, (self.kernel_size/2)-1, steps=int((self.kernel_size/2))) # computing only half of the window
        self.window_=0.54-0.46*torch.cos(2*math.pi*n_lin/self.kernel_size)


        # (1, kernel_size/2)
        n = (self.kernel_size - 1) / 2.0
        self.n_ = 2*math.pi*torch.arange(-n, 0).view(1, -1) / self.sample_rate # Due to symmetry, I only need half of the time axes

 


    def forward(self, waveforms):
        """
        Parameters
        ----------
        waveforms : `torch.Tensor` (batch_size, 1, n_samples)
            Batch of waveforms.
        Returns
        -------
        features : `torch.Tensor` (batch_size, out_channels, n_samples_out)
            Batch of sinc filters activations.
        """

        self.n_ = self.n_.to(waveforms.device)

        self.window_ = self.window_.to(waveforms.device)

        low = self.min_low_hz  + torch.abs(self.low_hz_)
        
        high = torch.clamp(low + self.min_band_hz + torch.abs(self.band_hz_),self.min_low_hz,self.sample_rate/2)
        band=(high-low)[:,0]
        
        f_times_t_low = torch.matmul(low, self.n_)
        f_times_t_high = torch.matmul(high, self.n_)

        band_pass_left=((torch.sin(f_times_t_high)-torch.sin(f_times_t_low))/(self.n_/2))*self.window_ # Equivalent of Eq.4 of the reference paper (SPEAKER RECOGNITION FROM RAW WAVEFORM WITH SINCNET). I just have expanded the sinc and simplified the terms. This way I avoid several useless computations. 
        band_pass_center = 2*band.view(-1,1)
        band_pass_right= torch.flip(band_pass_left,dims=[1])
        
        
        band_pass=torch.cat([band_pass_left,band_pass_center,band_pass_right],dim=1)

        
        band_pass = band_pass / (2*band[:,None])
        

        self.filters = (band_pass).view(
            self.out_channels, 1, self.kernel_size)

        return F.conv1d(waveforms, self.filters, stride=self.stride,
                        padding=self.padding, dilation=self.dilation,
                         bias=None, groups=1) 



class Encoder(nn.Module):
    def __init__(self,):
        super(Encoder,self).__init__()
        self.rate = 16000
        self.chunk_len = 400 #msec
        self.input_dim = int(self.rate*self.chunk_len/1000.00)
        pool_size = 3
        self.pool = nn.MaxPool1d(pool_size)        
        self.nonlinear = nn.LeakyReLU(0.2)
        self.lnorm_in = nn.LayerNorm(self.input_dim)
        current_input = self.input_dim
        #---------------------------- SINC CONV
        out_channels = 80
        kernel_size = 251
        self.sinc_conv = SincConv(out_channels, kernel_size, self.rate)
        current_input=int((current_input-kernel_size+1)/pool_size)
        self.lnorm_sinc = nn.LayerNorm([out_channels, current_input])
        #---------------------------- CONV 1
        in_channels = out_channels
        out_channels = 60
        kernel_size = 5        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size)
        current_input=int((current_input-kernel_size+1)/pool_size)
        self.lnorm1 = nn.LayerNorm([out_channels, current_input])        
        #---------------------------- CONV 2
        in_channels = out_channels
        out_channels = 60
        kernel_size = 5        
        self.conv2 = nn.Conv1d(in_channels, out_channels, kernel_size)
        current_input=int((current_input-kernel_size+1)/pool_size)
        self.lnorm2 = nn.LayerNorm([out_channels, current_input])        
        #------------------------------
        current_input = out_channels*current_input
        out_feats = 2048
        self.fc0 = nn.Linear(current_input, out_feats)
        self.bn0 = nn.BatchNorm1d(out_feats,momentum=0.05)
        #------------------------------
        current_input = out_feats
        out_feats = 1024
        self.fc1 = nn.Linear(current_input, out_feats)
        self.bn1 = nn.BatchNorm1d(out_feats,momentum=0.05)     

        
  

    def forward(self, x):
        #input: size  (batch, seq_len)
        batch=x.shape[0]
        seq_len=x.shape[1]
        x = self.lnorm_in(x)
        x = x.view(batch,1,seq_len)   #output: size  (batch, 1, seq_len)
        x = torch.abs(self.sinc_conv(x))
        x = self.nonlinear(self.lnorm_sinc(self.pool(x)))
        x = self.conv1(x)
        x = self.nonlinear(self.lnorm1(self.pool(x)))
        x = self.conv2(x)
        x = self.nonlinear(self.lnorm2(self.pool(x)))
        x = x.view(batch,-1)
        x = self.bn0(self.nonlinear(self.fc0(x)))
        x = self.bn1(self.nonlinear(self.fc1(x)))
        return x



class Discriminator(nn.Module):
    def __init__(self, in_features=1024, outputs=1):
        super(Discriminator, self).__init__()
        self.nonlinear = nn.LeakyReLU()
        self.linear1 = nn.Linear(in_features*2, in_features)
        self.linear2 = nn.Linear(in_features, outputs)
        #self.linear = nn.Linear(in_features*2, outputs)
        #nn.init.kaiming_normal_(self.linear.weight.data)
      
    def forward(self, x1, x2):
        x = torch.cat((x1,x2), dim=1)
        
        x = self.linear1(x)
        x = self.nonlinear(x)
        x = self.linear2(x)
        y = self.nonlinear(x)
        '''
        x = self.linear(x)
        y = self.nonlinear(x)
        '''

        return y


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.encoder = Encoder()
        self.discriminator = Discriminator()
      
    def forward(self, c_1, c_2, c_rnd):
        encoded_output1 = self.encoder(c_1)
        encoded_output2 = self.encoder(c_2)
        encoded_output_rnd = self.encoder(c_rnd)

        discriminator_output_pos = self.discriminator(encoded_output1, encoded_output2)
        discriminator_output_neg = self.discriminator(encoded_output1, encoded_output_rnd)

        return discriminator_output_pos, discriminator_output_neg

