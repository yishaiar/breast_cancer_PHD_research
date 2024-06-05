import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim




def show_and_save(img, file_name):
    r"""Show and save the image.

    Args:
        img (Tensor): The image.
        file_name (Str): The destination.

    """
    npimg = np.transpose(img.numpy(), (1, 2, 0))
    f = "./%s.png" % file_name
    plt.imshow(npimg, cmap='gray')
    plt.imsave(f, npimg)


def train(model, train_loader, epochs=20, lr=0.01,device='cpu'):
    r"""Train a RBM model.

    Args:
        model: The model.
        train_loader (DataLoader): The data loader.
        n_epochs (int, optional): The number of epochs. Defaults to 20.
        lr (Float, optional): The learning rate. Defaults to 0.01.

    Returns:
        The trained model.

    """
    # optimizer
    optimizer = optim.Adam(model.parameters(), lr)
    model.to(device)
    # train the RBM model
    
    
    model.train()  # Set model to training mode
    for epoch in range(epochs):
        epoch_loss = 0
        for i, [data, target] in enumerate(train_loader):
            optimizer.zero_grad()
            data = data.to(device)
            # target = target.to(device)
            reconstruct = model(data)
            
            loss = model.free_energy(data.view(data.size(0), -1)) - model.free_energy(reconstruct.view(reconstruct.size(0),-1))
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        print(f'Epoch {epoch+1}, Loss: {epoch_loss/len(train_loader)}')
        # print('Epoch %d\t Loss=%.4f' % (epoch, np.mean(loss_)))
    model.eval()  # Set model to evaluation mode after training
    return model
