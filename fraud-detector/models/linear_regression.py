import torch
import torch.nn as nn

class torchLinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)  


    def forward(self, x):
        return self.linear(x)





model = torchLinearRegression()
loss_fn = nn.MSELoss()
updater = torch.optim.SGD(model.parameters(), lr=0.05)

X = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]])
y = torch.tensor([[2.0], [4.0], [6.0], [8.0], [10.0]])

for epoch in range(1000):
    y_pred = model(X)
    loss = loss_fn(y_pred, y)
    updater.zero_grad()
    loss.backward()
    updater.step()

    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item():.4f}')

        

