class PaperSoccerEnv:


    def __init__(self,x_length = 8,y_length=10,gate_size: int =2):

        assert x_length % 2==0 and y_length % 2==0 and gate_size%2==0, "error"
        assert gate_size<y_length, "error"

        self.x_length = x_length
        self.y_length = y_length
        self.gate_size = gate_size
        self.cur_pos = [0,0] # starting pos

        self.used_paths = set()
        self.cur_player = -1 # -1(down) or 1(up)

        self.terminal_states = {-1:set(),1:set()}
        self.bouncing_places = set()

    def _create_edge(self,cur_pos,decision):


    def _create_boundaries(self):
        left_down_corner = (-self.x_length/2,-self.y_length/2)
        right_up_corner = (-self.x_length/2,self.y_length/2)
        for i in range(self.x_length):
            x = left_down_corner[0]
            y = left_down_corner[1]+i
            self.bouncing_places.add((x,y))
            self.bouncing_places.add((x+self.x_length,y))

        right_up_corner = (-self.x_length / 2, self.y_length / 2)
        for i in range(self.y_length):
            x = right_up_corner[0]+i
            y = right_up_corner[1]

            if abs(x)>=self.gate_size/2:
                self.bouncing_places.add((x,y))
                self.bouncing_places.add((x,y-self.y_length))



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



