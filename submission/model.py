import torch
import os
import numpy as np

# 1. The Supercharged 1-Feature Architecture
class NFBaseModel(torch.nn.Module):
    def __init__(self, num_channels, hidden_size=2048): 
        super(NFBaseModel, self).__init__()
        
        self.encoder = torch.nn.GRU(
            input_size=num_channels, 
            hidden_size=hidden_size, 
            num_layers=3, 
            batch_first=True,
            dropout=0.3 
        )
        self.output_layer = torch.nn.Linear(hidden_size, num_channels)

    def forward(self, x):
        output, _ = self.encoder(x)
        output = self.output_layer(output)
        return output

class Model:
    def __init__(self, monkey_name=""):
        self.monkey_name = monkey_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if self.monkey_name == 'beignet':
            self.num_channels = 89
        elif self.monkey_name == 'affi':
            self.num_channels = 239
        else:
            raise ValueError(f'No such a monkey: {self.monkey_name}')
        
        # Initialize TWO separate models for the ensemble
        self.net_mse = NFBaseModel(num_channels=self.num_channels, hidden_size=2048).to(self.device)
        self.net_huber = NFBaseModel(num_channels=self.num_channels, hidden_size=2048).to(self.device)
        self.average = None
        self.std = None

    def load(self, path=""):
        if not path:
            path = os.path.dirname(__file__)
            
        # Load MSE Weights
        filename_mse = f"model_{self.monkey_name}_mse.pth"
        full_path_mse = os.path.join(path, filename_mse)
        if os.path.exists(full_path_mse):
             self.net_mse.load_state_dict(torch.load(full_path_mse, map_location=self.device, weights_only=True))
        else:
            print(f"WARNING: Weights file {filename_mse} not found!")
            
        # Load Huber Weights
        filename_huber = f"model_{self.monkey_name}_huber.pth"
        full_path_huber = os.path.join(path, filename_huber)
        if os.path.exists(full_path_huber):
             self.net_huber.load_state_dict(torch.load(full_path_huber, map_location=self.device, weights_only=True))
        else:
            print(f"WARNING: Weights file {filename_huber} not found!")
            
        self.net_mse.eval()
        self.net_huber.eval()
        
        # Load Normalization Stats (Both models use the same stats)
        stats_filename = f"train_data_average_std_{self.monkey_name}.npz"
        stats_path = os.path.join(path, stats_filename)
        if os.path.exists(stats_path):
            stats = np.load(stats_path)
            self.average = stats['average']
            self.std = stats['std']

    def predict(self, X):
        n, t, c, f = X.shape
        
        # 1. Normalize
        if self.average is not None and self.std is not None:
            X_reshaped = X.reshape((n * t, -1))
            combine_max = self.average + 4 * self.std
            combine_min = self.average - 4 * self.std
            denom = combine_max - combine_min
            denom[denom == 0] = 1e-8
            
            X_norm = 2 * (X_reshaped - combine_min) / denom - 1
            X = X_norm.reshape((n, t, c, f))
            
        # 2. Extract ONLY feature 0 (LFP)
        x_input = X[:, :, :, 0]
        
        # 3. Mask future steps
        init_steps = 10
        future_steps = t - init_steps
        x_masked = np.concatenate([
            x_input[:, :init_steps, :], 
            np.repeat(x_input[:, init_steps-1:init_steps, :], future_steps, axis=1)
        ], axis=1)
        
        # 4. Predict with BOTH models and average the results
        x_tensor = torch.tensor(x_masked, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            pred_mse = self.net_mse(x_tensor).cpu().numpy()
            pred_huber = self.net_huber(x_tensor).cpu().numpy()
            
            # The Ensemble Average
            prediction_norm = (pred_mse + pred_huber) / 2.0
            
        # 5. Denormalize
        if self.average is not None and self.std is not None:
            dummy = np.zeros((n, t, c, f))
            dummy[:, :, :, 0] = prediction_norm
            dummy_reshaped = dummy.reshape((n * t, -1))
            
            dummy_denorm = (dummy_reshaped + 1) * denom / 2 + combine_min
            prediction = dummy_denorm.reshape((n, t, c, f))[:, :, :, 0]
        else:
            prediction = prediction_norm
            
        return prediction