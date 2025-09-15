
# This scripts contains the models for the data compression of AeroSense Surface pressure data.
import torch 
from torch import nn
from torchinfo import summary
from basic_tools import (
    single_conv1d_block, 
    single_trans_conv1d_block, 
    ConvBlock, 
    TransConvBlock
)
import math
import numpy as np
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import typing as tp
import sys
sys.path.append("../vector-quantize-pytorch")
sys.path.append("../vector-quantize-pytorch/vector-quantize-pytorch")
from vector_quantize_pytorch import ResidualVQ

VECTOR_QUANTIZER = False
device_ = 'cuda' if torch.cuda.is_available() else 'cpu'


""" 
This is the main class for the AutoEncoder. Many class/ blocks here were experimented on. Most of them are not used.

List with the most important ones:
    - Conv1d
    - ConvTranspose1d
    - ChannelwiseLinear
    - ResidualUnit
    - EncoderBlock_no_BN
    - Decoderblock
    - Discriminator
    - Encoder_ss00_no_BN
    - Decoder_ss00_prog 1 - 3
    - Autoencoder_modular
"""

class Conv1d(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.causal_padding = self.dilation[0] * (self.kernel_size[0] - 1)
        
        self.conv1d = nn.Conv1d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size[0],
            stride=self.stride[0],
            dilation=self.dilation[0],
            padding= 0,
            padding_mode='zeros' 
        )
    def forward(self, x):
        x = nn.functional.pad(x, (self.causal_padding//2, self.causal_padding-self.causal_padding//2), 'constant')
        return self.conv1d(x)


class Conv1d_sympad(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.causal_padding = self.dilation[0] * (self.kernel_size[0] - 1)
        self.conv1d = nn.Conv1d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size[0],
            stride=self.stride[0],
            dilation=self.dilation[0],
            padding= self.causal_padding//2,
            padding_mode='zeros'
        )
    def forward(self, x):
        return self.conv1d(x)

class Conv1d_2dsqueeze(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.causal_padding = self.dilation[0] * (self.kernel_size[0] - 1)
        self.conv2d_as_1d = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=(1,self.kernel_size[0]),
            stride=(1,self.stride[0]),
            padding=(0,self.causal_padding//2),
            padding_mode='zeros',
            dilation=(1,self.dilation[0])
        )

    def forward(self, x):
        x = x.unsqueeze(2)
        self.conv2d_as_1d(x)
        x = x.squeeze(2)
        return x
    
class Conv1d_for_encoder(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.causal_padding = self.dilation[0] * (self.kernel_size[0] - 1)
        self.conv1d = nn.Conv1d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size[0],
            stride=self.stride[0],
            dilation=self.dilation[0],
            padding=0,
            padding_mode='zeros'
        )

    def forward(self, x):
        # x = nn.functional.pad(x, (0,self.causal_padding), 'constant')
        # x  = nn.functional.pad(x, (self.causal_padding//2, self.causal_padding-self.causal_padding//2, 0, 0), 'constant')

        return self.conv1d(x)


class ConvTranspose1d(nn.ConvTranspose1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.causal_padding = self.dilation[0] * (self.kernel_size[0] - 1) + self.output_padding[0] - self.stride[0] * 2
        self.convtranspose1d = nn.ConvTranspose1d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size[0],
            stride=self.stride[0],
            dilation=self.dilation[0],
            padding=0,
            padding_mode='zeros'
        )
    def forward(self, x, output_size=None):
        if self.padding_mode != 'zeros':
            raise ValueError('Only `zeros` padding mode is supported for ConvTranspose1d')

        assert isinstance(self.padding, tuple)
        output_padding = self._output_padding(
            x, output_size, self.stride, self.padding, self.kernel_size, self.dilation)
        x = nn.functional.pad(x, (self.causal_padding//2, self.causal_padding - self.causal_padding//2))
        return self.convtranspose1d(x)


class ChannelWiseLinear(nn.Module):
    def __init__(self, in_channels, out_features):
        super(ChannelWiseLinear, self).__init__()
        self.in_channels = in_channels
        self.out_features = out_features
               
        self.linear_layers = nn.ModuleList([nn.Linear(out_features, out_features) for _ in range(in_channels)])
    def forward(self, x):
        batch_size, channels, size = x.size()
        x = x.view(batch_size, channels, size)

        outputs = [self.linear_layers[i](x[:, i, :]) for i in range(channels)]
        
        output = torch.stack(outputs, dim=1)
        
        return output



class ResidualUnit(nn.Module):
    def __init__(self, in_channels, out_channels, dilation):
        super().__init__()
        
        self.dilation = dilation

        self.layers = nn.Sequential(
            Conv1d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=5, dilation=dilation),
            nn.PReLU(),
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=1),
        )

    def forward(self, x):
        return x + self.layers(x)


class ResidualUnit_sympad(nn.Module):
    def __init__(self, in_channels, out_channels, dilation):
        super().__init__()
        
        self.dilation = dilation

        self.layers = nn.Sequential(
            Conv1d_sympad(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=5, dilation=dilation),
            nn.PReLU(),
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=1),
        )

    def forward(self, x):
        return x + self.layers(x)
    
class ResidualUnit_2dsqueeze(nn.Module):
    def __init__(self, in_channels, out_channels, dilation):
        super().__init__()
        
        self.dilation = dilation

        self.layers = nn.Sequential(
            Conv1d_2dsqueeze(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=5, dilation=dilation),
            nn.PReLU(),
            nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(1,1)
            )
        )

    def forward(self, x):
        y = x.squeeze(2)
        y = self.layers(x)
        y = y.unsqueeze(2)
        return x + y
    
    
class ResidualUnitDynamicPadding(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layers = nn.Sequential(
            Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, dilation=1, padding=0),
            nn.PReLU()
        )

    def forward(self, x):
        output = self.layers(x)
        if x.size(-1) < output.size(-1):  # Pad the input if needed
            pad_size = output.size(-1) - x.size(-1)
            x = F.pad(x, (0, pad_size))
        return x + output

    

class EncoderBlock_no_BN_sym(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride):
        super().__init__()

        self.layers = nn.Sequential(
            ResidualUnit_sympad(in_channels=int(in_channels),
                         out_channels=int(in_channels), dilation=3),
            nn.PReLU(),
            Conv1d_sympad(in_channels=int(in_channels), out_channels=int(out_channels), 
            kernel_size=kernel, stride=stride),
        )

    def forward(self, x):
        return self.layers(x)
    
class EncoderBlock_no_BN(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()

        self.layers = nn.Sequential(
            ResidualUnit(in_channels=int(in_channels),
                         out_channels=int(in_channels), dilation=3),
            nn.PReLU(),
            Conv1d(in_channels=int(in_channels), out_channels=int(out_channels), 
            kernel_size=2*stride, stride=stride),
        )

    def forward(self, x):
        return self.layers(x)


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()

        self.layers = nn.Sequential(
            ResidualUnit(in_channels=int(in_channels),
                         out_channels=int(in_channels), dilation=9),
            nn.BatchNorm1d(int(in_channels)), # < ---------------------
            nn.PReLU(), 
            Conv1d(in_channels=int(in_channels), out_channels=int(out_channels), 
            kernel_size=2*stride, stride=stride),

            nn.BatchNorm1d(int(out_channels)), # < ---------------------
        )

    def forward(self, x):
        return self.layers(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()

        self.layers = nn.Sequential(
            ConvTranspose1d(in_channels=int(in_channels), out_channels=int(out_channels), 
            kernel_size=2*stride, stride=stride),
            # nn.BatchNorm1d(int(out_channels)),
            nn.ELU(), #nn.PReLU(),
            ResidualUnit(in_channels=int(out_channels), out_channels=int(out_channels),
                         dilation=1),
            # nn.BatchNorm1d(int(out_channels)),
            nn.ELU(), #nn.PReLU(),
            ResidualUnit(in_channels=int(out_channels), out_channels=int(out_channels),
                         dilation=3),
            nn.ELU(), #nn.PReLU(),
            ResidualUnit(in_channels=int(out_channels), out_channels=int(out_channels),
                         dilation=9),

        )

    def forward(self, x):
        return self.layers(x)


class Discriminator(nn.Module): # using simple Discriminator first to see how good it is, might expand to full GAN

    def __init__(self, input_channels = 36, seq_len = 800):
        super().__init__()
        self.seq_len = seq_len

        self.layers = nn.Sequential(
            nn.Conv1d(in_channels=input_channels, out_channels=64, kernel_size=15, stride=1, padding=7), # general feature extraction, no special reason for parameters
            nn.LeakyReLU(0.2),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=7, stride=2, padding=3),
            nn.LeakyReLU(0.2),
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=7, stride=2, padding=3),
            nn.LeakyReLU(0.2),
            nn.Conv1d(in_channels=256, out_channels=512, kernel_size=7, stride=2, padding=3),
            nn.LeakyReLU(0.2),

            nn.AdaptiveAvgPool1d(1),    # since I am taking SoundStream as a example, using avgPool
            nn.Flatten(),

            nn.Linear(512,1),
            
        )

    def forward(self,x):
        return self.layers(x)



class Encoder_ss00_no_BN(nn.Module):
    def __init__(self, C, D):
        super().__init__()
        self.layers = nn.Sequential(
            Conv1d(in_channels=36, out_channels=C, kernel_size=7),
            nn.PReLU(),
            EncoderBlock_no_BN(in_channels=C ,out_channels=24, stride=1),
            nn.PReLU(),
            EncoderBlock_no_BN(in_channels=24 ,out_channels=20, stride=2),
            nn.PReLU(),
            EncoderBlock_no_BN(in_channels=20 ,out_channels=16, stride=2),
            nn.PReLU(),
            EncoderBlock_no_BN(in_channels=16 ,out_channels=12, stride=2),
            nn.PReLU(),
            Conv1d(in_channels=12, out_channels=D, kernel_size=1), 
        )

    def forward(self, x):
        return self.layers(x)

class Encoder_ss00_no_BN_sym(nn.Module):
    def __init__(self, C, D):
        super().__init__()
        self.layers = nn.Sequential(
            Conv1d(in_channels=36, out_channels=C, kernel_size=7),
            nn.PReLU(),
            EncoderBlock_no_BN(in_channels=C ,out_channels=24, kernel=1, stride =1),
            nn.PReLU(),
            EncoderBlock_no_BN(in_channels=24 ,out_channels=20, kernel=3, stride =2),
            nn.PReLU(),
            EncoderBlock_no_BN(in_channels=20 ,out_channels=16, kernel=3, stride =2),
            nn.PReLU(),
            EncoderBlock_no_BN(in_channels=16 ,out_channels=12, kernel=3, stride =2),
            nn.PReLU(),
            Conv1d(in_channels=12, out_channels=D, kernel_size=1), 

        )

    def forward(self, x):
        return self.layers(x)

class Encoder_ss00(nn.Module):
    def __init__(self, C, D):
        super().__init__()
        self.layers = nn.Sequential(
            Conv1d(in_channels=36, out_channels=C, kernel_size=7),
            nn.BatchNorm1d(C), # < ---------------------
            nn.PReLU(),
            EncoderBlock(in_channels=C ,out_channels=24, stride=1),
            nn.PReLU(),
            EncoderBlock(in_channels=24 ,out_channels=20, stride=2),
            nn.PReLU(),
            EncoderBlock(in_channels=20 ,out_channels=16, stride=2),
            nn.PReLU(),
            EncoderBlock(in_channels=16 ,out_channels=12, stride=2),
            nn.PReLU(),
            Conv1d(in_channels=12, out_channels=D, kernel_size=1), 

        )

    def forward(self, x):
        return self.layers(x)

class Decoder_ss00_1(nn.Module): # slightly altered Decoder like proposed in the previous BA thesis (Activations)
    def __init__(self, C, D):
        super().__init__()

        self.layers = nn.Sequential(

            ChannelWiseLinear(9, 97),
            nn.ELU(), 
            Conv1d(in_channels=D, out_channels=12, kernel_size=1), 
            nn.ELU(), 
            DecoderBlock(in_channels=12, out_channels=16, stride=2),
            nn.ELU(), 
            DecoderBlock(in_channels=16, out_channels=20, stride=2),
            nn.ELU(), 
            DecoderBlock(in_channels=20, out_channels=24, stride=2),
            nn.ELU(),
            DecoderBlock(in_channels=24, out_channels=36, stride=1),
            nn.ELU(), 
            Conv1d(in_channels=C, out_channels=36, kernel_size=3)
        )
    
    def forward(self, x): # this is for new residual units, so I have an output of 800
        output = self.layers(x)
        target_size = 800
        current_size = output.size(-1)
        padding_needed = target_size - current_size

        output_padded = nn.functional.pad(output, (0, padding_needed))
        return output_padded

class Decoder_ss00_2(nn.Module):    # Slightly modified with a Channel Wise linear layer
    def __init__(self, C, D):
        super().__init__()
        self.mult_factors=[3/2, 6/5, 5/4, 4/3]


        self.layers = nn.Sequential(

            ChannelWiseLinear(9, 100),
            nn.ELU(), #nn.PReLU(),

            Conv1d(in_channels=D, out_channels=12, kernel_size=1), 
            nn.ELU(), #nn.PReLU(),

            DecoderBlock(in_channels=12, out_channels=16, stride=2),
            nn.ELU(), #nn.PReLU(),
            
            DecoderBlock(in_channels=16, out_channels=20, stride=2),
            ChannelWiseLinear(20,400),
            nn.ELU(), #nn.PReLU(),
            nn.Dropout(0.5),

            DecoderBlock(in_channels=20 ,out_channels=24, stride=2),
            nn.ELU(), #nn.PReLU(),

            DecoderBlock(in_channels=24 ,out_channels=36, stride=1),         
            nn.ELU(), #nn.PReLU(),
            Conv1d(in_channels=C, out_channels=36, kernel_size=3)
        )
    
    def forward(self, x):
        return self.layers(x)
    

class Decoder_ss00_3(nn.Module): # Complex Decoder
    def __init__(self, C, D):
        super().__init__()
        """
        Decoder with explicit channel and data size progression.
        Args:
            channel_progression (list): List of channel sizes, e.g., [9, 12, 15, 18, 22, 26, 30, 36].
            data_size_progression (list): List of strides to increase data size, e.g., [1, 2, 2, 2, 1, 1, 1].
        """
        super().__init__()
        channel_progression = [9, 12, 15, 18, 22, 26, 30, 36]
        data_size_progression = [1, 2, 2, 2, 1, 1, 1]

        assert len(channel_progression) - 1 == len(data_size_progression), \
            "The number of data size strides must match the number of channel transitions."

        layers = []
        layers.append(ChannelWiseLinear(9, 100))
        layers.append(nn.ELU())
        layers.append(Conv1d(in_channels=D, out_channels=12, kernel_size=1))
        layers.append(nn.ELU())

        for i in range(1,len(data_size_progression)):
            layers.append(
                DecoderBlock(
                    in_channels=channel_progression[i],
                    out_channels=channel_progression[i + 1],
                    stride=data_size_progression[i]
                )
            )
            if channel_progression[i] == 15:  # Maybe there is a nicer way to do this, but works for now
                layers.append(ChannelWiseLinear(18,400))
                layers.append(nn.ELU())
                layers.append(nn.Dropout(0.5))

        layers.append(Conv1d(channel_progression[-1], 36, kernel_size=3))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)



class AutoEncoder(nn.Module): 

    def __init__(self, arch_id: str, c_in: int = 36, c_factor: int = 4,
                 RVQ: bool = True, codebook_size: int = 1024, quantizers=8, dec_code: int = 1, enc_code: int = 1): 
        super().__init__()
        """
        This is the main class of the AutoEncoder.
        
        Args:
            arch_id (string): ss00 or aa00. Only select ss00, aa00 was just a debugging model
            c_in, c_factor (int): only used for aa00 debugging model
            RVQ (bool): default True. Gives the choice of using RVQ
            codebook_size (int): default 1024. Number of codewords per quantizer
            quantizers (int): default 8. Number of quantizers
            dec_code (int): default 1. Choose between the 3 different decoders. Values [1,2,3]
            enc_code (int): default 1. Choose between using batchnorm or not. Values [0,1]. For GVSOC should always be 1
        """
        self.arch_id = arch_id 
        self.c_in = c_in
        self.c_out = int(c_in / c_factor)
        self.c_factor = c_factor
        self.enc_code = enc_code
        self.dec_code = dec_code
        self.encoder = self.get_encoder()
        self.decoder = self.get_decoder()
        self.RVQ = RVQ
        self.codebook_size = codebook_size
        self.quantizers = quantizers
        
        if self.RVQ == True:
            self.quantizer = ResidualVQ(
                            dim = 100,
                            num_quantizers = self.quantizers,                     # specify number of quantizers, do 97 when no padding, and 100 else
                            codebook_size = self.codebook_size, # codebook size
                            threshold_ema_dead_code = 2,     
                            kmeans_init = True,      # set to True
                            kmeans_iters = 100,      # number of kmeans iterations to calculate the centroids for the codebook on init
                            )

    def forward(self, x: torch.Tensor): 
        x = self.encoder(x)
        if self.RVQ == True:
            x, indices, commit_loss = self.quantizer(x)
        x = self.decoder(x)

        if self.RVQ == True:
            return x, indices, commit_loss
        else:
            return x
    
    def get_encoder(self): 
        """
        Returns the encoder of the AutoEncoder. 

        Parameters
        ----------
        arch_id : str
            Architecture ID of the AutoEncoder. 
        """
        
        match self.arch_id:

            case "aa00": 
                return Encoder_aa00(c_in = self.c_in, c_factor = self.c_factor, sequetial_len = 2)
            case "ss00":
                if self.enc_code == 0:
                    return Encoder_ss00(C=36,D=9)
                elif self.enc_code == 1:
                    print("-"*50)
                    print("NO BN NO IDENTITY")
                    print("-"*50)
                    return Encoder_ss00_no_BN(C=36,D=9)
                else:
                    print("Encoder specifics not found.")
            case _: 
                print("Architecture ID not found.")
                
    def get_rvq(self):
        print("RVQ")
        return self.quantizer

    def get_decoder(self):
        """
        Returns the decoder of the AutoEncoder. 

        Parameters
        ----------
        arch_id : str
            Architecture ID of the AutoEncoder. 
        """
        
        match self.arch_id:

            case "aa00": 
                return Decoder_aa00(c_in = self.c_out, c_factor = self.c_factor, sequetial_len = 2)
            case "ss00":
                if self.dec_code == 1:
                    print("-"*50)
                    print("DECODER 1")
                    print("-"*50)
                    return Decoder_ss00_1(C=36, D=9)  # Classic Decoder architecture
                elif self.dec_code == 2:
                    print("-"*50)
                    print("DECODER 2")
                    print("-"*50)
                    return Decoder_ss00_2(C=36, D=9) # Decoder with ChannelWiseLinear
                elif self.dec_code == 3:
                    print("-"*50)
                    print("DECODER 3")
                    print("-"*50)
                    return Decoder_ss00_3(C=36, D=9) # Complex Decoder
                
            case _: 
                print("Architecture ID not found.")


class Encoder_aa00(nn.Module):
    
    def __init__(self, c_in, c_factor: int = 4, sequetial_len: int = 3): 
        super().__init__()
        self.c_in = c_in
        cout = self.c_in
        self.sequetial_len = sequetial_len

        model: tp.List[nn.Module] = []

        model += [ConvBlock(c_in = int(cout), c_out = int(cout / 2), 
                                kernel_size_residual = 3, kernel_size_down_sampling = 7, 
                                stride_in = 1, strid_down_sampling = 2, padding_mode=str('replicate'), layers = self.sequetial_len)] 
        for _ in range(2, c_factor, 2):
            cout = cout / 2
            model += [ConvBlock(c_in = int(cout), c_out = int(cout / 2), 
                                kernel_size_residual = 3, kernel_size_down_sampling = 7, 
                                stride_in = 1, strid_down_sampling = 2, padding_mode=str('replicate'), layers = self.sequetial_len)]

        model += [ConvBlock(c_in = int(cout / 2), c_out = int(cout / 2), 
                                kernel_size_residual = 3, kernel_size_down_sampling = 7, 
                                stride_in = 1, strid_down_sampling = 2, padding_mode=str('replicate'), layers = self.sequetial_len)]

        self.conv = nn.Sequential(*model)

    def forward(self, x: torch.Tensor): 
        x = self.conv(x)
        return x    
    
class Decoder_aa00(nn.Module): 

    def __init__(self, c_in, c_factor: int = 4, sequetial_len: int = 3): 
        super().__init__()
        self.c_in = c_in
        cout = self.c_in
        self.sequetial_len = sequetial_len

        model: tp.List[nn.Module] = []
        
        model += [TransConvBlock(
            c_in=int(c_in), c_out=int(c_in), 
            kernel_size=3, kernel_size_up_sampling=4, 
            stride_residual=1, stride_up_sampling=2, 
            padding=1, output_padding=0, layers = self.sequetial_len)]
        
        model += [TransConvBlock(
            c_in=int(c_in), c_out=int(cout*2), 
            kernel_size=3, kernel_size_up_sampling=4, 
            stride_residual=1, stride_up_sampling=2, 
            padding=1, output_padding=0, layers = self.sequetial_len)] 
        
        for _ in range(2, c_factor, 2):
            cout = cout * 2
            model += [TransConvBlock(
                c_in=int(cout), c_out=int(cout*2), 
                kernel_size=3, kernel_size_up_sampling=4, 
                stride_residual=1, stride_up_sampling=2, 
                padding=1, output_padding=0, layers = self.sequetial_len)] 
            
        model += [nn.Conv1d(in_channels = int(cout*2), out_channels = int(cout*2), 
                            kernel_size = 7, stride = 1, padding = 3)]

        self.deconv = nn.Sequential(*model)
        
    def forward(self, x: torch.Tensor): 
        x = self.deconv(x)
        return x




if __name__ == "__main__": 

    device_ = 'cuda' if torch.cuda.is_available() else 'cpu'
    rand_input1 = torch.rand(1, 36, 800) #.to(device_)

    model  = AutoEncoder(arch_id = "ss00", RVQ=True, dec_code = 3, enc_code=1).quantizer #.to(device_)
    summary(model, input_size = (1, 9, 100), verbose = 1, depth = 5)


