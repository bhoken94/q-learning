import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch as T

class LinearClassifier(nn.Module):
    def __init__(self, lr , n_classes, input_dims):
        super(LinearClassifier, self).__init__()
        self.fc1 = nn.Linear(*input_dims, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, n_classes)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        # Funzione per il calcolo della loss. Questo tipo di loss è molto usato per problemi di classificazione ML
        self.loss = nn.CrossEntropyLoss()

        # Impostiamo quale device utilizzare (GPU o CPU)
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        # Inviamo la rete alla GPU o CPU
        self.to(self.device)

    def forward(self, data):
        x = F.sigmoid(self.fc1(data))
        x = F.sigmoid(self.fc2(x))
        output = self.fc3(x)

        return output

    def learn(self, data, labels):
        # Azzeriamo i gradienti dell'optimizer
        self.optimizer.zero_grad()
        # Convertiamo in tensor
        data = T.tensor(data).to(self.device)
        labels = T.tensor(labels).to(self.device)
        predictions = self.forward(data)

        # Calcoliamo quanto sono lontane le previsioni dalle labels
        cost = self.loss(predictions, labels)

        # Facciamo Backpropagation del costo
        cost.backward()

        self.optimizer.step()