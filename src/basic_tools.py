# This scripts contains the basic tools for the data compression of AeroSense Surface pressure data.

import torch 
from torch import nn 






####################################################################################################################################################################
############################################## BASIC CNN BLOCKS FOR META PAPER ##################################################################################### 
####################################################################################################################################################################

def single_conv1d_block(in_channels: int, out_channel: int, kernel_size: int, stride: int, padding: int, padding_mode:str, *args, **kwargs) -> nn.Sequential:
    """ 
    Implementing a single convolutional layer with batch normalization and ELU activation.

    Parameters
    ----------
    in_channels : int
        Number of input channels. In the input layer this is equal to the number of the sensors used. 
    out_channel : int   
        Number of output channels. Modyfing this parameter effect the spatial relation between the sensors.   
    kernel_size : int
        Kernel size of the convolutional layer. 
    stride : int
        Stride of the convolutional layer. Choose it with care, because it affects the dimension of the output.
    padding : str
        Padding of the convolutional layer. Choose it with care, because it affects the dimension of the output.
    *args :
        Variable length argument list.
    **kwargs :
        Arbitrary keyword arguments.
    """

    return nn.Sequential( 
            nn.Conv1d(in_channels, out_channel, kernel_size, stride, padding, padding_mode = padding_mode, *args, **kwargs), 
            nn.BatchNorm1d(out_channel), 
            nn.ELU()
        )


def single_trans_conv1d_block(in_channels: int, out_channel: int, kernel_size: int, stride: int, padding: int, output_padding: int, *args, **kwargs) -> nn.Sequential:
    """ 
    Implementing a single Trans convolutional layer with batch normalization and ELU activation.

    Parameters
    ----------
    in_channels : int
        Number of input channels.  
    out_channel : int   
        Number of output channels.   
    kernel_size : int
        Kernel size of the convolutional layer. 
    stride : int
        Stride of the convolutional layer. Choose it with care, because it affects the dimension of the output.
    padding : str
        Padding of the convolutional layer. Choose it with care, because it affects the dimension of the output.
    *args :
        Variable length argument list.
    **kwargs :
        Arbitrary keyword arguments.
    """
    return nn.Sequential(
            nn.ConvTranspose1d(in_channels, out_channel, kernel_size, stride, padding, output_padding, *args, **kwargs),
            nn.BatchNorm1d(out_channel),
            nn.ELU()
        )

class ConvBlock(nn.Module): 
    """ 
    Implementing a convolutional block with residual connections and downsampling. 

    The ConVBlock is inspired by the architecture in
    "High Fidelity Neural Audio Compression" by Alexandre Defossez et al. (2023)
    
    """
    def __init__(self, c_in: int, c_out: int, kernel_size_residual: int, kernel_size_down_sampling: int,stride_in: int, strid_down_sampling: int, padding_mode:str, layers:int): 
        
        """
        Initialize the ConvBlock class.

        Parameters
        ----------
        c_in : int 
            Number of input channels.
        c_out : int
            Number of output channels.
        kernel_size_residual : int
            Kernel size of the residual block in the backbone of the block.
        kernel_size_down_sampling : int
            Kernel size of the downsampling block
        stride_in : int
            Stride of the residual block in the backbone of the block.
        strid_down_sampling : int
            Stride of the downsampling block.
        padding : str   
            padding of the convolutional layers.
        """    
        
        super().__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.kernel_size = kernel_size_residual
        self.stride = stride_in
        self.kernel_down_sampling = kernel_size_down_sampling
        self.stride_down_sampling = strid_down_sampling
        self.padding_mode = padding_mode

        assert layers > 0
        models = []
        for _ in range(layers):
            models.append(single_conv1d_block(in_channels = self.c_in , out_channel = self.c_in, kernel_size = self.kernel_size, stride = self.stride, padding= 1, padding_mode=self.padding_mode))
        self.backbone = nn.Sequential(*models)
        
        self.downsampling = nn.Conv1d(in_channels = self.c_in, out_channels = c_out, kernel_size = self.kernel_down_sampling, stride = self.stride_down_sampling, padding = 3, padding_mode = self.padding_mode)

    
    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        """
        Propagate the input through the ConvBlock.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        """
        
        res = x
        x = self.backbone(x)
        x = x + res
        x = self.downsampling(x)
        
        return x


class TransConvBlock(nn.Module):
    """ 
    Implementing a Transose convolutional block with residual connections and Upsampling for the Decoder part. 

    The ConVBlock is inspired by the architecture in
    "High Fidelity Neural Audio Compression" by Alexandre Defossez et al. (2023)
    
    """
    
    def __init__(self, 
                 c_in: int, 
                 c_out: int, 
                 kernel_size: int, 
                 kernel_size_up_sampling: int, 
                 stride_residual: int, 
                 stride_up_sampling: int, 
                 padding: int, 
                 output_padding: int,
                 layers:int, 
                 *args, **kwargs): 
        
        """ 
        Initialize the Transpose ConvBlock class.

        Parammeter 
        ----------

        c_in : int
            Number of input channels.
        c_out : int
            Number of output channels.
        kernel_size : int
            Kernel size of the residual block in the backbone of the block.
        kernel_size_up_sampling : int
            Kernel size of the upsampling block
        stride_residual : int  
            Stride of the residual block in the backbone of the block.
        stride_up_sampling : int
            Stride of the upsampling block.
        padding : str
            padding of the convolutional layers.
        output_padding : int
            Output padding of the convolutional layers.
        *args :
            Variable length argument list.
        **kwargs :
            Arbitrary keyword arguments.
        """
        
        
        super().__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.kernel_size = kernel_size
        self.kernel_size_up_sampling = kernel_size_up_sampling
        self.stride_residual = stride_residual
        self.stride_up_sampling = stride_up_sampling
        self.padding = padding
        self.output_padding = output_padding

        assert layers > 0
        models = []
        for _ in range(layers):
            models.append(single_trans_conv1d_block(in_channels=self.c_in, out_channel=self.c_in, kernel_size=self.kernel_size, stride=self.stride_residual, padding=self.padding, output_padding=self.output_padding))
        self.backbone = nn.Sequential(*models)

        self.upsampling = nn.ConvTranspose1d(in_channels=self.c_in, out_channels=self.c_out, kernel_size=self.kernel_size_up_sampling, stride=self.stride_up_sampling, padding=self.padding, output_padding=self.output_padding)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Propagate the input through the Transpose ConvBlock.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        """

        res = x 
        x = self.backbone(x)
        x = x + res 
        x = self.upsampling(x)
        return x





