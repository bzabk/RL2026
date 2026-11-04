import numpy as np

class PaperSoccerEnv:
    def __init__(self,x_length = 8,y_length=10,gate_size: int =2):

        assert x_length % 2==0 and y_length % 2==0 and gate_size%2==0, "error"
        assert gate_size<y_length, "error"


        self.grid = np.zeros((y_length+3,x_length+1,9),dtype=np.int8)

        # channels 0-7 represent actions
        # 8: is boundary

        self.x_length = x_length
        self.y_length = y_length
        self.gate_size = gate_size
        self.cur_pos = [(y_length+3)//2,(x_length+1)//2] # starting pos

        self._create_boundaries()


    def _grid_overview(self):
        print(self.grid[:,:,8])

    def _create_boundaries(self):

        left = self.x_length//2-self.gate_size//2
        right = self.x_length//2+self.gate_size//2
        self.grid[0,:,8]=1
        self.grid[-1,:,8]=1

        self.grid[:,0,8]=1
        self.grid[:,-1,8]=1

        self.grid[[1,-2],:left+1,8]=1
        self.grid[[1,-2],right:,8]=1



    def _create_terminal_states(self):
        # winning states for player -1
        for i in range(-self.gate_size/2,self.gate_size/2+1,1):
            self.terminal_states.add(i,self.y_length/2+1)

        # winning states for player 1
        for i in range(-self.gate_size/2,self.gate_size/2+1,1):
            self.terminal_states.add(i,-self.y_length/2+1)


    def reset(self):
        pass


    def step(self,action):
        # possible actions: {1,2,3,4,5,6,7,8,9}
        #  1           2              3
        #
        #
        #
        # 8      starting_point       4
        #
        #
        # 7            6              5

        match action:
            case 1:
                edge_begin = self.cur_pos.copy()
                self.cur_pos[0]-=1
                self.cur_pos[1]+=1
                edge_end = self.cur_pos.copy()
                edge = frozenset([edge_begin,edge_end])
                self.used_paths.add(edge)
            case 2:
                self.cur_pos[1]+=1
            case 3:
                self.cur_pos[0]+=1
                self.cur_pos[1]+=1
            case 4:
                self.cur_pos[0]+=1
            case 5:
                self.cur_pos[0]+=1
                self.cur_pos[1]-=1
            case 6:
                self.cur_pos[1]-=1
            case 7:
                self.cur_pos[0]-=1
                self.cur_pos[1]-=1
            case 8:
                self.cur_pos[0]-=1



        # own goal cur player loses
        if self.cur_pos in self.terminal_states[self.cur_player]:
            return False

        # opponent loses
        if self.cur_pos in self.terminal_states[self.cur_player*(-1)]:
            return False
        # check bouncing
        if self.cur_pos in self.bouncing_places:
            pass


if __name__ == "__main__":
    env = PaperSoccerEnv(x_length=8,y_length=10,gate_size=2)
    env._grid_overview()


