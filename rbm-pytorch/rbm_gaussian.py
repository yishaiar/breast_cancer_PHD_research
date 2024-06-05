



import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class RBM(nn.Module):
    def __init__(self, n_visible, n_hidden,output_shape,k=1,binary = True):
        super(RBM, self).__init__()
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.k = k
        
        self.output_shape = output_shape
        
        # Initialize weights and biases
        self.W = nn.Parameter(torch.randn(n_hidden, n_visible) * 0.1)
        self.v_bias = nn.Parameter(torch.zeros(n_visible))
        self.h_bias = nn.Parameter(torch.zeros(n_hidden))
        
        if binary:
            self.h_to_v = self.h_to_v_bernoulli
            self.free_energy = self.free_energy_bernoulli
            self.forward = self.forward_bernoulli
        else:
            self.h_to_v = self.h_to_v_gaussian_bernoulli
            self.free_energy = self.free_energy_gaussian_bernoulli
            self.forward = self.forward_gaussian_bernoulli
            

            
        
    def sample_from_p(self, p):
        return torch.bernoulli(p)
    
    def v_to_h(self, v):
        p_h_given_v = torch.sigmoid(F.linear(v, self.W, self.h_bias))
        return p_h_given_v, self.sample_from_p(p_h_given_v)
    
    def h_to_v_bernoulli(self, h):
        p_v_given_h = torch.sigmoid(F.linear(h, self.W.t(), self.v_bias))
        return p_v_given_h, self.sample_from_p(p_v_given_h)
    def h_to_v_gaussian_bernoulli(self, h):
        mean_v_given_h = F.linear(h, self.W.t(), self.v_bias)
        return mean_v_given_h, torch.sigmoid(torch.normal(mean_v_given_h, 1.0)) #torch.normal(mean_v_given_h, 1.0)
    
    def free_energy_bernoulli(self, v):
        vbias_term = v.mv(self.v_bias)
        wx_b = F.linear(v, self.W, self.h_bias)
        hidden_term = wx_b.exp().add(1).log().sum(1)
        return (-hidden_term - vbias_term).mean()
    def free_energy_gaussian_bernoulli(self, v):
        vbias_term = 0.5 * torch.sum((v - self.v_bias) ** 2, dim=1)
        wx_b = F.linear(v, self.W, self.h_bias)
        hidden_term = wx_b.exp().add(1).log().sum(1)
        return (-hidden_term + vbias_term).mean()
    def forward_bernoulli(self, v):
        # s = v.size()
        
        v = v.view(-1, self.n_visible)
        v = v.bernoulli()  # Binarize the input data

        # Positive phase
        ph0, h0 = self.v_to_h(v)
        
        # Negative phase
        vk = v
        for _ in range(self.k):
            _, hk = self.v_to_h(vk)
            _, vk = self.h_to_v(hk)

        return vk.view(-1, *self.output_shape)
    def forward_gaussian_bernoulli(self, v, k=1):
        v = v.view(-1, self.n_visible)

        # Positive phase
        ph0, h0 = self.v_to_h(v)
        
        # Negative phase
        vk = v
        for _ in range(self.k):
            _, hk = self.v_to_h(vk)
            mean_v_given_h, vk = self.h_to_v(hk)

        return vk.view(-1, *self.output_shape)
# def train(model, train_loader, epochs=10, lr=0.01, device='cpu'):
#     model.to(device)
#     # optimizer = optim.SGD(model.parameters(), lr=lr)
#     optimizer = optim.Adam(model.parameters(), lr)

#     model.train()  # Set model to training mode
#     for epoch in range(epochs):
#         epoch_loss = 0
#         for i, [data, target] in enumerate(train_loader):
#             optimizer.zero_grad()
#             data = data.to(device)
#             reconstruct = model(data)
            
#             loss = model.free_energy(data.view(data.size(0), -1)) - model.free_energy(reconstruct.view(reconstruct.size(0),-1))
#             loss.backward()
#             optimizer.step()
            
#             epoch_loss += loss.item()
        
#         print(f'Epoch {epoch+1}, Loss: {epoch_loss/len(train_loader)}')
#     return model
# Example usage:
# Assuming `train_dataset` is a PyTorch dataset with images flattened to 784 features
# from torch.utils.data import DataLoader
# train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
# rbm = RBM(n_visible=784, n_hidden=256)
# train(rbm, train_loader, epochs=10, lr=0.01, k=1, device='cuda')
