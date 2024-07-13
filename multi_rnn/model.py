import torch
import torch.nn as nn


class MiltiRegionRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, connectome):
        super(MiltiRegionRNN, self).__init__()
        self.C = connectome
        self.num_regions, _ = self.C.shape
        self.weights_ih = nn.Parameter(torch.randn(self.num_regions, input_size, hidden_size))
        self.weights_hh = nn.Parameter(torch.randn(self.num_regions, hidden_size, hidden_size))
        self.weight_rhh = nn.Parameter(torch.randn(self.num_regions, hidden_size, hidden_size))
        self.bias = nn.Parameter(torch.randn(self.num_regions, hidden_size))
        self.read_out = nn.Linear(self.num_regions* hidden_size, output_size)
        self.hidden_size = hidden_size

        # Initialize weights
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.weights_ih)
        nn.init.xavier_uniform_(self.weights_hh)
        nn.init.kaiming_uniform_(self.weight_rhh, nonlinearity='tanh')

    def forward(self, x):
        device = x.device
        seq_len, bs, _ = x.shape
        H_0 = torch.zeros(self.num_regions, bs, self.hidden_size, device=device)
        H_history = []

        for t in range(seq_len):
            x_t = x[t, :, :].unsqueeze(0).repeat(self.num_regions, 1, 1)
            H_t = torch.bmm(x_t, self.weights_ih) + torch.bmm(H_0, self.weights_hh) + self.bias[:, None]

            # Correcting the shape and broadcasting for the recurrent connection weighted sum
            C_expanded = self.C.unsqueeze(2).unsqueeze(3)  # Shape: (num_regions, num_regions, 1, 1)
            H_0_expanded = H_0.unsqueeze(1)  # Shape: (num_regions, 1, bs, hidden_size)
            
            # Perform element-wise multiplication and sum across regions
            recurrent_sum = (C_expanded * H_0_expanded).sum(dim=0)  # Summing along the appropriate dimension (regions)
            
            H_t += torch.bmm(recurrent_sum, self.weight_rhh)

            H_t = torch.tanh(H_t)
            H_history.append(H_t)
            H_0 = H_t

        H_history = torch.stack(H_history).transpose(1, 2).contiguous()
        H_history = H_history.view(seq_len, bs, -1)
        output = self.read_out(H_history)
        return output
    
    def lesion_forward(self, x, region_idx: tuple):
        weights_ih_original = self.weights_ih.detach().clone()
        weights_hh_original = self.weights_hh.detach().clone()
        weight_rhh_original = self.weight_rhh.detach().clone()
        bias_original = self.bias.detach().clone()
        C_original = self.C.detach().clone()

        self.weights_ih[region_idx, :, :] = 0
        self.weights_hh[region_idx, :, :] = 0
        self.weight_rhh[region_idx, :, :] = 0
        self.bias[region_idx, :] = 0
        self.C[region_idx, :] = 0
        self.C[:, region_idx] = 0

        output = self.forward(x)

        self.weights_ih = nn.Parameter(weights_ih_original)
        self.weights_hh = nn.Parameter(weights_hh_original)
        self.weight_rhh = nn.Parameter(weight_rhh_original)
        self.bias = nn.Parameter(bias_original)
        self.C = C_original

        return output