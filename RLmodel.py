import itertools 
import numpy as np 
from monte_carlo import MCTS


class RL:
    
    def __init__(self, res_net, nonres_net, env, args):
        
        self.res_net=res_net
        self.nonres_net=nonres_net
        self.env=env 
        self.args=args 
        
    def get_res_training_samples(self):
        
        board, player=self.env.get_initial_state()
        boards, players, policies=[], [], []
        mcts=MCTS(self.res_net, self.env, self.args)
        
        for i in itertools.count():
            
            actions, probs=mcts.simulate(board, player)
            if sum(probs)==0:
                print('current i failed')
                continue
            probs=probs/sum(probs)
            policy=np.zeros(self.args['RowNum']*self.args['ColNum'])
            policy[actions]=probs
            boards.append(board)
            players.append(player)
            policies.append(policy)
            action=np.random.choice(actions, p=probs)
            next_board, next_player=self.env.next_state(board, action, player)
            
            winner=self.env.is_terminal_state(next_board, action, player)
            if winner is not None:
                values=[]
                if winner==0:
                    values=np.zeros((len(players),1))
                else:
                    for player in players:
                        if player==winner:
                            values.append(np.array([1]))
                        else:
                            values.append(np.array([-1]))
                return boards, players, policies, values 
            
            board=next_board
            player=next_player
            
    def get_nonres_training_samples(self):
        
        board, player=self.env.get_initial_state()
        boards, players, policies=[], [], []
        mcts=MCTS(self.nonres_net, self.env, self.args)
        
        for i in itertools.count():
            
            actions, probs=mcts.simulate(board, player)
            if sum(probs)==0:
                print('current i failed')
                continue
            probs=probs/sum(probs)
            policy=np.zeros(self.args['RowNum']*self.args['ColNum'])
            policy[actions]=probs
            boards.append(board)
            players.append(player)
            policies.append(policy)
            action=np.random.choice(actions, p=probs)
            next_board, next_player=self.env.next_state(board, action, player)
            
            winner=self.env.is_terminal_state(next_board, action, player)
            if winner is not None:
                values=[]
                if winner==0:
                    values=np.zeros((len(players),1))
                else:
                    for player in players:
                        if player==winner:
                            values.append(np.array([1]))
                        else:
                            values.append(np.array([-1]))
                return boards, players, policies, values 
            
            board=next_board
            player=next_player
            
    def trainRL(self):
        policyloss_res_list=[]
        valueloss_res_list=[]
        policyloss_nonres_list=[]
        valueloss_nonres_list=[]
        step_list=[]
        
        for n in itertools.count():
            
            print('training step='+'%s' %n)
            
            res_boards, res_players, res_policies, res_values=self.get_res_training_samples()
            self.res_net.train(res_boards, res_players, res_policies, res_values)
            policyloss_res, valueloss_res=self.res_net.get_loss(res_boards, res_players, res_policies, res_values)
            
            nonres_boards, nonres_players, nonres_policies, nonres_values=self.get_nonres_training_samples()
            self.nonres_net.train(nonres_boards, nonres_players, nonres_policies, nonres_values)
            policyloss_nonres, valueloss_nonres=self.nonres_net.get_loss(nonres_boards, nonres_players, nonres_policies, nonres_values)
            
            filename='%s'%n
            
            policyloss_res_list.append(policyloss_res)
            valueloss_res_list.append(valueloss_res)
            policyloss_nonres_list.append(policyloss_nonres)
            valueloss_nonres_list.append(valueloss_nonres)
            step_list.append(n)
            
            np.save('./saved_models/policyloss_res_list', policyloss_res_list)
            np.save('./saved_models/valueloss_res_list', valueloss_res_list)
            np.save('./saved_models/policyloss_nonres_list', policyloss_nonres_list)
            np.save('./saved_models/valueloss_nonres_list', valueloss_nonres_list)
            np.save('./saved_models/step_list', step_list)
            
            self.res_net.save_model(filename)
            self.nonres_net.save_model(filename)
            
            
            
            
            
        
        
        
        
            
            
            

                    
                
            
            
            
            
            
            
            
            
        
        
        
