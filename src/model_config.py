from dataclasses import dataclass


@dataclass
class ModelConfig:
        model_id: str = "0000:0000:0000:0000"
        arch_id: str = "aa00"
        window_size: int = 800
        latent_channels: int = 36
        latent_seq_len: int = 400
        train_time: float = 0.0 
        train_loss: int = 0
        valid_loss: int = 0 
        alpha: float = 0.9
        wind_speed: int = 10
        batch_size: int = 64
        epochs: int = 80
        description: str = "short description of the model"