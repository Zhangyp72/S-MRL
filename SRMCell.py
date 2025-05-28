import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import norse.torch as norse


class SRMCell(nn.Module):
    """
    SRM（Spike Response Model）
    """
    def __init__(self, input_size, hidden_size, tau_m=10.0, tau_s=5.0):
        super(SRMCell, self).__init__()
        self.hidden_size = hidden_size
        self.tau_m = tau_m
        self.tau_s = tau_s

        
        self.w = nn.Linear(input_size, hidden_size)
        self.u = nn.Linear(hidden_size, hidden_size)

        
        self.reset_state()

    def reset_state(self):
        
        self.v_mem = None  
        self.psp = None  

    def forward(self, x, prev_state=None):
        
        if prev_state is None:
            batch_size = x.shape[0]
            self.v_mem = torch.zeros(batch_size, self.hidden_size, device=x.device)
            self.psp = torch.zeros(batch_size, self.hidden_size, device=x.device)
        else:
            self.v_mem, self.psp = prev_state

        
        self.psp = (1 - 1/self.tau_s) * self.psp + self.w(x) + self.u(self.v_mem)

       
        self.v_mem = (1 - 1/self.tau_m) * self.v_mem + self.psp

        
        spikes = (self.v_mem > 1.0).float()

        
        self.v_mem = self.v_mem * (1 - spikes)

        return spikes, (self.v_mem, self.psp)



input_size = 64*64
hidden_size = 256
batch_size = 1

srm_cell = SRMCell(input_size, hidden_size)
test_input = torch.rand((batch_size, input_size))  

spike_output, _ = srm_cell(test_input)
print(spike_output.shape)  
