import numpy as np
import torch 
import torch.nn as nn 
from torch.autograd import Variable
import torch.nn.functional as F

class PolicyNet(nn.Module):
    
    def __init__(self, args):
        
        super(PolicyNet, self).__init__()
        self.args=args
        self.padding=int((self.args['conv_kernel_size']-1)/2)
        
        self.F_Conv_Block1=nn.Conv2d(self.args['conv_num_filters'], self.args['conv_num_filters'], self.args['conv_kernel_size'], 1, self.padding)
        self.S_Conv_Block1=nn.Conv2d(self.args['conv_num_filters'], self.args['conv_num_filters'], self.args['conv_kernel_size'], 1, self.padding)
        
        self.F_Conv_Block2=nn.Conv2d(self.args['conv_num_filters'], self.args['conv_num_filters'], self.args['conv_kernel_size'], 1, self.padding)
        self.S_Conv_Block2=nn.Conv2d(self.args['conv_num_filters'], self.args['conv_num_filters'], self.args['conv_kernel_size'], 1, self.padding)
        
        self.final_Conv=nn.Conv2d(self.args['conv_num_filters'], 1, 3, 1, 0)
        self.final_Linear=nn.Linear((self.args['RowNum']-2)*(self.args['ColNum']-2), self.args['RowNum']*self.args['ColNum'])
        
    def forward(self, curinput):
        
        x=curinput
        
        tempx=x
        x=self.F_Conv_Block1(x)
        x=nn.BatchNorm2d(self.args['conv_num_filters'])(x)
        x=F.relu(x)
        x=self.S_Conv_Block1(x)
        x=nn.BatchNorm2d(self.args['conv_num_filters'])(x)
        x+=tempx
        x=F.relu(x)
        
        tempx=x
        x=self.F_Conv_Block2(x)
        x=nn.BatchNorm2d(self.args['conv_num_filters'])(x)
        x=F.relu(x)
        x=self.S_Conv_Block2(x)
        x=nn.BatchNorm2d(self.args['conv_num_filters'])(x)
        x+=tempx
        x=F.relu(x)
        
        policy=self.final_Conv(x)
        policy=nn.BatchNorm2d(1)(policy)
        policy=F.relu(policy)
        policy=nn.Flatten()(policy)
        policy=self.final_Linear(policy)
        policy=F.softmax(policy,dim=1)
        return policy
    
    
class ValueNet(nn.Module):
    
    def __init__(self, args):
        
        super(ValueNet, self).__init__()
        self.args=args
        self.padding=int((self.args['conv_kernel_size']-1)/2)
        
        self.F_Conv_Block1=nn.Conv2d(self.args['conv_num_filters'], self.args['conv_num_filters'], self.args['conv_kernel_size'], 1, self.padding)
        self.S_Conv_Block1=nn.Conv2d(self.args['conv_num_filters'], self.args['conv_num_filters'], self.args['conv_kernel_size'], 1, self.padding)
        
        self.F_Conv_Block2=nn.Conv2d(self.args['conv_num_filters'], self.args['conv_num_filters'], self.args['conv_kernel_size'], 1, self.padding)
        self.S_Conv_Block2=nn.Conv2d(self.args['conv_num_filters'], self.args['conv_num_filters'], self.args['conv_kernel_size'], 1, self.padding)
        
        self.final_Conv=nn.Conv2d(self.args['conv_num_filters'], 1, 3, 1, 0)
        self.final_Linear=nn.Linear((self.args['RowNum']-2)*(self.args['ColNum']-2), 1)
        
    def forward(self, curinput):
        
        x=curinput
        
        tempx=x
        x=self.F_Conv_Block1(x)
        x=nn.BatchNorm2d(self.args['conv_num_filters'])(x)
        x=F.relu(x)
        x=self.S_Conv_Block1(x)
        x=nn.BatchNorm2d(self.args['conv_num_filters'])(x)
        x+=tempx
        x=F.relu(x)
        
        tempx=x
        x=self.F_Conv_Block2(x)
        x=nn.BatchNorm2d(self.args['conv_num_filters'])(x)
        x=F.relu(x)
        x=self.S_Conv_Block2(x)
        x=nn.BatchNorm2d(self.args['conv_num_filters'])(x)
        x+=tempx
        x=F.relu(x)
        
        value=self.final_Conv(x)
        value=nn.BatchNorm2d(1)(value)
        value=F.relu(value)
        value=nn.Flatten()(value)
        value=self.final_Linear(value)
        value=torch.tanh(value)
        return value  

    
class GomokuResNet(object):
    
    def __init__(self, env, args):
        
        self.args=args
        self.env=env
        self.policy_net=PolicyNet(args)
        self.value_net=ValueNet(args)
        self.policy_optimizer=torch.optim.Adam(params=self.policy_net.parameters(), lr=self.args['lr'], betas=(0.9, 0.999))
        self.value_optimizer=torch.optim.Adam(params=self.value_net.parameters(), lr=self.args['lr'], betas=(0.9, 0.999))
        
    def predict(self, chess, player):
        
        states=np.zeros((1, 3, self.args['RowNum'], self.args['ColNum']))
        states[0]=self.fit_transform(chess, player)
        states=torch.FloatTensor(states)
        policy=self.policy_net(states)
        value=self.value_net(states)
        return policy.detach().numpy().copy()[0], value.detach().numpy().copy()[0]
    
    def train(self, boards, players, policies, values):
        
        states=np.zeros((len(players), 3, self.args['RowNum'], self.args['ColNum']))
        for k in range(len(players)):
            states[k]=self.fit_transform(boards[k], players[k])
        states=Variable(torch.FloatTensor(states))
        
        policies_output=self.policy_net(states)
        values_output=self.value_net(states)
        
        policies_input=Variable(torch.FloatTensor(policies))
        values_input=Variable(torch.FloatTensor(values))
        
        batch_size=len(players)
        
        value_loss=0
        for k in range(batch_size):
            value_loss+=(values_input[k][0]-values_output[k][0])*(values_input[k][0]-values_output[k][0])
        value_loss=0.5*value_loss/batch_size
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()
        
        policy_loss=0
        for k1 in range(batch_size):
            local_loss=0
            for k2 in range(self.args['RowNum']*self.args['ColNum']):
                local_loss+=-policies_input[k1][k2]*torch.log(policies_output[k1][k2])
            policy_loss+=local_loss
        policy_loss=policy_loss/batch_size
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
    def get_loss(self, boards, players, policies, values):
        
        states=np.zeros((len(players), 3, self.args['RowNum'], self.args['ColNum']))
        for k in range(len(players)):
            states[k]=self.fit_transform(boards[k], players[k])
        states=Variable(torch.FloatTensor(states))
        
        policies_output=self.policy_net(states)
        values_output=self.value_net(states)
        
        policies_input=Variable(torch.FloatTensor(policies))
        values_input=Variable(torch.FloatTensor(values))
        
        batch_size=len(players)
        
        value_loss=0
        for k in range(batch_size):
            value_loss+=(values_input[k][0]-values_output[k][0])*(values_input[k][0]-values_output[k][0])
        value_loss=0.5*value_loss/batch_size
       
        policy_loss=0
        for k1 in range(batch_size):
            local_loss=0
            for k2 in range(self.args['RowNum']*self.args['ColNum']):
                local_loss+=-policies_input[k1][k2]*torch.log(policies_output[k1][k2])
            policy_loss+=local_loss
        policy_loss=policy_loss/batch_size
        
        return policy_loss.detach().numpy(), value_loss.detach().numpy()
        
        
        
    
    def save_model(self, filename):
        
        policy_file='./saved_models/Res_policy_net_'+filename+'.pkl'
        value_file='./saved_models/Res_value_net_'+filename+'.pkl'
        torch.save(self.policy_net.state_dict(), policy_file)
        torch.save(self.value_net.state_dict(), value_file)
        
    def load_model(self, filename):
        
        policy_file='./saved_models/Res_policy_net_'+filename+'.pkl'
        value_file='./saved_models/Res_value_net_'+filename+'.pkl'
        self.policy_net.load_state_dict(torch.load(policy_file))
        self.value_net.load_state_dict(torch.load(value_file))
        
    def fit_transform(self, board, player):
        
        def transform(board, player):
            transboard=np.zeros((self.args['RowNum'], self.args['ColNum']))
            for row in range(self.args['RowNum']):
                for col in range(self.args['ColNum']):
                    if board[row][col]==player:
                        transboard[row][col]=1
            return transboard
        
        feature=np.zeros((3, self.args['RowNum'], self.args['ColNum']))
        if player==1:
            feature[2]=np.ones((self.args['RowNum'], self.args['ColNum']))
        newboard=board.copy()
        if player==-1:
            for row in range(self.args['RowNum']):
                for col in range(self.args['ColNum']):
                    if newboard[row][col]!=0:
                        newboard[row][col]*=-1
        feature[0]=transform(newboard, 1)
        feature[1]=transform(newboard, -1)
        return feature
        
            
        


            
        
        
        
        
        
        
        
        
        
        
    
    
    

        
        
        
        
        
        
        
    
        
        
        
        
    

