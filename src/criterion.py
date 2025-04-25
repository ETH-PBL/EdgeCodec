import torch 
import torch.nn as nn


""" 
This class is used to construct and choose different possible the loss function
for the training and validation of the model.

We have the following options for the loss function:
    - MSE Loss <https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html#torch.nn.MSELoss>
    - L1 Loss <https://pytorch.org/docs/stable/generated/torch.nn.L1Loss.html#torch.nn.L1Loss>
    - Huber Loss <https://pytorch.org/docs/stable/generated/torch.nn.HuberLoss.html#torch.nn.HuberLoss>
    - Smooth L1 Loss <https://pytorch.org/docs/stable/generated/torch.nn.SmoothL1Loss.html#torch.nn.SmoothL1Loss>
    - Mixed (combination of two given loss functions)

params criterion_method: str -> the name of the loss function to be used. It can be one of the following:  
    - mse
    - l1
    - huber
    - smooth_l1
params reduction_type: str -> the type of the reduction to be used. It can be one of the following:
    - mean
    - sum
    - none
params criterion_type: str -> the type of the loss function to be used. It can be one of the following:
    - single -> only one loss function is used
    - mixed -> two loss functions are used
params delta: float -> the delta parameter for the Huber and Smooth L1 Loss functions. 
params alpha: float -> the alpha parameter for the mixed loss function.
params *args: -> additional arguments for the loss functions.
params **kwargs: -> additional keyword arguments for the loss functions.

"""
class ReconstructionLoss(nn.Module):
    
    def __init__(
            self, 
            reduction_type: str,
            alpha: float = 0.5,
            *args, **kwargs) -> None:    
        
        super().__init__(*args, **kwargs)
        
        self.delta = kwargs.get("delta", 0.02)
        self.alpha = alpha
        self.criterion_type = kwargs.get("criterion_type", "mixed")

        self.reduction = reduction_type
      
        self.mse_criterion = nn.MSELoss(reduction=self.reduction)
        self.l1_criterion = nn.L1Loss(reduction=self.reduction)  
        self.hubber_criterion = nn.HuberLoss(reduction=self.reduction, delta= self.delta)
        self.smooth_l1_criterion = nn.SmoothL1Loss(reduction=self.reduction, beta=self.delta)

    def criterion_selector(self, x, y, criterion_method) -> nn.Module:
        match criterion_method: 
            case"mse" : 
                criterion =  self.mse_criterion(x, y)
            case "l1":
                criterion = self.l1_criterion(x, y)
            case "huber":
                criterion = self.hubber_criterion(x, y)
            case "smooth_l1":
                criterion = self.smooth_l1_criterion(x, y)
            case _:
                raise ValueError("The given criterion method is not supported")
            
        return criterion
    

    


class ReconstructionLossSingle(ReconstructionLoss): 
    def __init__(self, 
            criterion_method: tuple , 
            reduction_type: str,
            *args, **kwargs) -> None:

        super().__init__( reduction_type, *args, **kwargs)  
        self.criterion_method_0 = criterion_method

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        print(self.criterion_method_0)
        criterion = self.criterion_selector(x, y, self.criterion_method_0)

        return criterion
        

class ReconstructionLossMixed(ReconstructionLoss): 
    def __init__(self, criterion_method: tuple, reduction_type: str, alpha, *args, **kwargs) -> None:
        super().__init__(reduction_type, alpha, *args, **kwargs)
        self.criterion_method_0 = criterion_method[0]
        self.criterion_method_1 = criterion_method[1]

    def forward(self, x:torch.Tensor, y:torch.Tensor) -> torch.Tensor: 

        criterion_0 = self.criterion_selector(x, y, self.criterion_method_0)
        criterion_1 = self.criterion_selector(x, y, self.criterion_method_1)

        x_fft = torch.fft.fft(x, dim=2)
        y_fft = torch.fft.fft(y, dim=2)

        fft_loss = torch.sum((x_fft - y_fft).abs()**2)

        criterion = self.alpha * criterion_0 + (1 - self.alpha) * criterion_1

        return criterion


if __name__ == "__main__": 
    
    # Test the reconstruction loss function
    x = torch.randn(10, 30, 100)
    y = torch.randn(10, 30, 100)

    criterion_method = ("mse", "huber")
    reduction_type = "mean"
    criterion_type = "single"
    delta = 1.0
    alpha_ = 0.5

    # criterion = ReconstructionLossSingle(criterion_method, reduction_type)
    criterion = ReconstructionLossMixed(criterion_method, reduction_type, alpha = 0.7)
    print(criterion(x, y))