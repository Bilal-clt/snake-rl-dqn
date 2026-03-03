import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(Linear_QNet, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size1)
        self.linear2 = nn.Linear(hidden_size1, hidden_size2)
        self.linear3 = nn.Linear(hidden_size2, output_size)
        #super() calls the constructor of the parent class nn.Module
        #nn.Linear creates a linear layer (y=Ax+b)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x
    #x initially is tensor of state(state is list)
    #state = tensor([s1, s2, ..., s15]) after self.linear1(x),
    #x = tensor([h1, h2, h3, ..., h256])
    #F.relu is the activation function. like some tensor elements might be less than 0,
    #so it makes it zero

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)
        #state_dict() contains the parameters of the model
        #eg: weights and biases
        #torch.save saves the model to the specified file
        #os.path.join joins directory and file name into a full path
        #os.makedirs creates the directory if it doesn't exist
    
class QTrainer:
    def __init__(self, model, target_model, lr, gamma):
        self.model = model
        self.target_model = target_model
        self.lr = lr
        self.gamma = gamma
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.HuberLoss() #errror function
        #optim.Adam optimizes the model parameters - eg: weights and biases
        #nn.MSELoss computes the mean squared error between 
        #the predicted and target values of Q-values(immediate+future reward) of actions

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        #convert inputs to tensors

        if len(state.shape) == 1:
            # reshape to (1, x)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )
        #if the input is a single sample, reshape it to have a batch dimension
        #target = reward + gamma * max(next_predicted Q value) -> only do this if not done
        # 1: predicted Q values with current state
        pred = self.model(state)
        #pred is tensor of predicted Q-values(s,a) of actions for each action
        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                with torch.no_grad():
                    Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))
            target[idx][torch.argmax(action[idx]).item()] = Q_new
            #It does NOT mean “how many times it played”
        #idx means “how many samples I’m training on right now”
        #hit wall => means done is true
        #Only add future Q-value if the agent didn’t die”
        #torch.max() returns the value
        #torch.max(tensor([2.3, 0.1, 0.2])) = 2.3
        #compute the target Q values using the Bellman equation


        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        #loss = mean(target - pred)^2
        loss.backward()
        #Compute gradients
        #Tells each weight:
        #“You contributed this much to the error”
        self.optimizer.step()
        #optimizer updates weights & biases
        #model becomes better
        #zero the gradients, compute the loss, backpropagate, and update the model parameters
    