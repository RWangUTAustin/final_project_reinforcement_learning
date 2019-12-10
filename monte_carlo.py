import math 
import numpy as np 

## this section of code is mainly the reuse of [11]

class MCTS:
    
    def __init__(self, net, env, args):
        
        self.net=net
        self.env=env
        self.args=args
        
        self.visit_count = {}  
        self.mean_action_value = {}  
        self.prior_probability = {}  
        self.terminal_state = {}
        self.total_visit_count = {}
        self.available_actions = {}
        
    def getboardstr(self, board):
        finalstr=''
        for k1 in range(self.args['RowNum']):
            for k2 in range(self.args['ColNum']):
                if board[k1][k2]!=0:
                    playerstr='%s' %int(board[k1][k2])
                    k1str='%s' %int(k1)
                    k2str='%s' %int(k2)
                    finalstr=finalstr+playerstr+'('+k1str+','+k2str+')#'
        return finalstr
    
    def simulate(self, board, player):
        
        for k in range(self.args['search_num']):
            self.search(board, player)
        board_str=self.getboardstr(board)
        return self.available_actions[board_str].copy(), self.visit_count[board_str].copy()
    
    def search(self, board, player):
        
        board_str=self.getboardstr(board)
        if board_str not in self.prior_probability:
            return -self.expand(board, player)
        index=self.select(board_str)
        action=self.available_actions[board_str][index]
        next_board, next_player=self.env.next_state(board, action, player)
        next_board_str=self.getboardstr(next_board)
        if next_board_str not in self.terminal_state:
            self.terminal_state[next_board_str]=self.env.is_terminal_state(next_board, action, player)
        value=0
        if self.terminal_state[next_board_str] is None:
            value=self.search(next_board, next_player)
        elif self.terminal_state[next_board_str]==player:
            value=1
        self.backup(board_str, index, value)
        return -value
    
    def select(self, board_str):
        
        max_value=-math.inf
        best_index=None 
        for i in range(len(self.available_actions[board_str])):
            cur_value = self.args['c_puct'] * self.prior_probability[board_str][i] * math.sqrt(
                self.total_visit_count[board_str]) / (1.0 + self.visit_count[board_str][i])
            cur_value += self.mean_action_value[board_str][i]
            if cur_value > max_value:
                max_value=cur_value 
                best_index=i
        return best_index
    
    def backup(self, board_str, index, value):
        
        self.mean_action_value[board_str][index] = (self.mean_action_value[board_str][index] * self.visit_count[board_str][
            index] + value) / (self.visit_count[board_str][index] + 1.0)
        self.visit_count[board_str][index] += 1
        self.total_visit_count[board_str] += 1
        
    def expand(self, board, player):
        
        policy, value=self.net.predict(board, player)
        board_str=self.getboardstr(board)
        actions=self.env.available_actions(board)
        self.available_actions[board_str]=actions 
        self.prior_probability[board_str] = policy[actions] / sum(policy[actions])
        self.total_visit_count[board_str] = 1
        self.mean_action_value[board_str] = np.zeros(len(actions))
        self.visit_count[board_str] = np.zeros(len(actions))
        return value
        
        
        
        
        
        
            
        
        
        
        
        