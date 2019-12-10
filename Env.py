import numpy as np 


class GomokuEnv(object):
    
    def __init__(self, args):
        self.args=args
        
    def get_initial_state(self):
        chess=np.zeros((self.args['RowNum'], self.args['ColNum']))
        return chess, 1
      
    def next_player(self, player):
        if player==0:
            return None
        if player==1:
            return -1
        if player==-1:
            return 1
        
    def next_state(self, chess, action, player):
        curx=action//self.args['ColNum']
        cury=action%self.args['ColNum']
        newchess=chess.copy()
        newchess[curx][cury]=player
        return newchess, self.next_player(player)
    
    def is_terminal_state(self, chess, action, player):
        count=0
        for k1 in range(self.args['RowNum']):
            for k2 in range(self.args['ColNum']):
                if chess[k1][k2]==1 or chess[k1][k2]==-1:
                    count+=1
        if count==self.args['ColNum']*self.args['RowNum']:
            return 0
        if count==0:
            return None
        directions=[[1,1],[1,-1], [1,0], [0,1]]
        for direction in directions:
            if self.is_win(action, player, direction, chess):
                return player
        return None
    
    def is_win(self, action, player, direction, chess):
        curx=action//self.args['ColNum']
        cury=action%self.args['ColNum']
        count=0
        if 0<=curx<self.args['RowNum'] and 0<=cury<self.args['ColNum'] and chess[curx][cury]==player:
            count=1
        nextx=curx+direction[0]
        nexty=cury+direction[1]
        while 0<=nextx<self.args['RowNum'] and 0<=nexty<self.args['ColNum'] and chess[nextx][nexty]==player:
            count+=1
            nextx+=direction[0]
            nexty+=direction[1]
        nextx=curx-direction[0]
        nexty=cury-direction[1]
        while 0<=nextx<self.args['RowNum'] and 0<=nexty<self.args['ColNum'] and chess[nextx][nexty]==player:
            count+=1
            nextx-=direction[0]
            nexty-=direction[1]
        return count>=self.args['TileNum_Win']
    
    def available_actions(self, chess):
        result=[]
        for row in range(self.args['RowNum']):
            for col in range(self.args['ColNum']):
                if chess[row][col]==0:
                    result.append(row*self.args['ColNum']+col)
        return result
        
            
        
    
    
    
    
         
    
    

    
    