import matplotlib.pyplot as plt
import data_handler as dh
from model import MLP
from torch import nn
import torch


def train(model, x, y, x_val, y_val, optimizer, criterion, epochs, device):
    loss_train = []
    loss_test= []
    max_loss = 1000
    for epoch in range(epochs):
        losses = 0
        for i in range(x.shape[0]):
            model.to(device)

            optimizer.zero_grad()

            output = model.forward(x[i])
            #print(output.shape)
            #print(y[i].shape)
            loss = criterion(output, y[i])
            loss.backward()

            optimizer.step()

            losses += loss.item()

        if (epoch+1) % 2 == 0:
            loss_train.append(loss.item())
            test_losses = 0
            
            with torch.no_grad():
                model.eval()
                for j in range(x_val.shape[0]):
                    output = model.forward(x_val[j])
                    loss = criterion(output, y_val[j])
                    test_losses += loss.item()

            if test_losses/x_val.shape[0] < max_loss:
                max_loss = test_losses
                torch.save(model.state_dict(), 'store_sales.pth')

            loss_test.append(loss.item())
                
            model.train()
                
            print("Epoch: {}/{} \t Train Loss: {}, Test Loss: {}".format(epoch + 1, epochs, losses/x.shape[0], test_losses/x_val.shape[0]))
    
    plt.plot(loss_train)
    plt.plot(loss_test)
    plt.show()
x_train, x_test, y_train, y_test =  dh.load_data("final_train.csv", batch_size=100, shuffle = True)
model = MLP(13,[10,8], 1)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train(model, x_train, y_train, x_test, y_test, optimizer, criterion= criterion, epochs=200, device = device)