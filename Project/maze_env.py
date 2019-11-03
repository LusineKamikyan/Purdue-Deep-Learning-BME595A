import PIL
import numpy as np
import time
import sys
if sys.version_info.major == 2:
    import Tkinter as tk
    import Tkinter.PhotoImage 
    from Tkinter import *
else:
    import tkinter as tk
import random

UNIT = 40   # pixels
MAZE_H = 4  # grid height
MAZE_W = 4  # grid width


class Maze(tk.Tk, object):
    def __init__(self):
        super(Maze, self).__init__()
        self.action_space = ['u', 'd', 'l', 'r']
        self.n_actions = len(self.action_space)
        self.title('maze')
        
        ##################
        
        #ws = self.winfo_screenwidth() # width of the screen
        #hs = self.winfo_screenheight() # height of the screen
        #x = (ws/2) - (MAZE_W * UNIT/2)
        #y = (hs/2) - (MAZE_H * UNIT/2)
        
        self.geometry('%dx%d+%d+%d' % (MAZE_W * UNIT, MAZE_H * UNIT, 0, 0))
        
        self._build_maze()

    def _build_maze(self):
        self.canvas = tk.Canvas(self, bg='black',
                           height=MAZE_H * UNIT,
                           width=MAZE_W * UNIT)
        
        # create grids
        for c in range(0, MAZE_W * UNIT, UNIT):
            x0, y0, x1, y1 = c, 0, c, MAZE_H * UNIT
            self.canvas.create_line(x0, y0, x1, y1)
        for r in range(0, MAZE_H * UNIT, UNIT):
            x0, y0, x1, y1 = 0, r, MAZE_H * UNIT, r
            self.canvas.create_line(x0, y0, x1, y1)

        # create origin
        origin = np.array([20, 20])

        obs1_center = origin + np.array([UNIT * 2, UNIT*1])
        self.obs1 = self.canvas.create_rectangle(
            obs1_center[0] - 15, obs1_center[1] - 15,
            obs1_center[0] + 15, obs1_center[1] + 15,
            fill='blue')
        obs2_center = origin + np.array([UNIT*1, UNIT * 2]) # random.randint(2,3)])
        self.obs2 = self.canvas.create_rectangle(
            obs2_center[0] - 15, obs2_center[1] - 15,
            obs2_center[0] + 15, obs2_center[1] + 15,
            fill='blue')
    
        
#        obs3_center = origin + np.array([UNIT*2, UNIT * 3])
#        self.obs3 = self.canvas.create_rectangle(
#            obs3_center[0] - 15, obs3_center[1] - 15,
#            obs3_center[0] + 15, obs3_center[1] + 15,
#            fill='blue')
        
        ghost_center = origin + np.array([UNIT*3, UNIT * 3])
        self.ghost = self.canvas.create_arc(
                ghost_center[0] - 15, ghost_center[1] - 15,
                ghost_center[0] + 15, ghost_center[1] + 15,
                fill='orange', start=-45, extent=270)

        oval_center01 = origin + np.array([UNIT*0, UNIT * 1])
        self.oval01 = self.canvas.create_oval(
            oval_center01[0] - 15, oval_center01[1] - 15,
            oval_center01[0] + 15, oval_center01[1] + 15,
            fill='yellow')
        
#        oval_center02 = origin + np.array([UNIT*0, UNIT * 2])
#        self.oval02 = self.canvas.create_oval(
#            oval_center02[0] - 15, oval_center02[1] - 15,
#            oval_center02[0] + 15, oval_center02[1] + 15,
#            fill='yellow')
#        oval_center03 = origin + np.array([UNIT*0, UNIT * 3])
#        self.oval03 = self.canvas.create_oval(
#            oval_center03[0] - 15, oval_center03[1] - 15,
#            oval_center03[0] + 15, oval_center03[1] + 15,
#            fill='yellow')
#        oval_center04 = origin + np.array([UNIT*0, UNIT * 4])
#        self.oval04 = self.canvas.create_oval(
#            oval_center04[0] - 15, oval_center04[1] - 15,
#            oval_center04[0] + 15, oval_center04[1] + 15,
#            fill='yellow')
#
#        oval_center05 = origin + np.array([UNIT*0, UNIT *5])
#        self.oval05 = self.canvas.create_oval(
#            oval_center05[0] - 15, oval_center05[1] - 15,
#            oval_center05[0] + 15, oval_center05[1] + 15,
#            fill='yellow')
        
#        oval_center10 = origin + np.array([UNIT*1, UNIT * 0])
#        self.oval10 = self.canvas.create_oval(
#            oval_center10[0] - 15, oval_center10[1] - 15,
#            oval_center10[0] + 15, oval_center10[1] + 15,
#            fill='yellow')
        
#        oval_center11 = origin + np.array([UNIT*1, UNIT * 1])
#        self.oval11 = self.canvas.create_oval(
#            oval_center11[0] - 15, oval_center11[1] - 15,
#            oval_center11[0] + 15, oval_center11[1] + 15,
#            fill='yellow')
#        
#        oval_center12 = origin + np.array([UNIT*1, UNIT * 2])
#        self.oval12 = self.canvas.create_oval(
#            oval_center12[0] - 15, oval_center12[1] - 15,
#            oval_center12[0] + 15, oval_center12[1] + 15,
#            fill='yellow')
        
        oval_center13 = origin + np.array([UNIT*1, UNIT * 3])
        self.oval13 = self.canvas.create_oval(
            oval_center13[0] - 15, oval_center13[1] - 15,
            oval_center13[0] + 15, oval_center13[1] + 15,
            fill='yellow')
        
#        oval_center14 = origin + np.array([UNIT*1, UNIT * 4])
#        self.oval14 = self.canvas.create_oval(
#            oval_center14[0] - 15, oval_center14[1] - 15,
#            oval_center14[0] + 15, oval_center14[1] + 15,
#            fill='yellow')
#        
#        oval_center15 = origin + np.array([UNIT*1, UNIT * 5])
#        self.oval15 = self.canvas.create_oval(
#            oval_center15[0] - 15, oval_center15[1] - 15,
#            oval_center15[0] + 15, oval_center15[1] + 15,
#            fill='yellow')
#        
#        oval_center20 = origin + np.array([UNIT*2, UNIT * 0])
#        self.oval20 = self.canvas.create_oval(
#            oval_center20[0] - 15, oval_center20[1] - 15,
#            oval_center20[0] + 15, oval_center20[1] + 15,
#            fill='yellow')
#        oval_center21 = origin + np.array([UNIT*2, UNIT * 1])
#        self.oval21 = self.canvas.create_oval(
#            oval_center21[0] - 15, oval_center21[1] - 15,
#            oval_center21[0] + 15, oval_center21[1] + 15,
#            fill='yellow')
#        oval_center23 = origin + np.array([UNIT*2, UNIT * 3])
#        self.oval23 = self.canvas.create_oval(
#            oval_center23[0] - 15, oval_center23[1] - 15,
#            oval_center23[0] + 15, oval_center23[1] + 15,
#            fill='yellow')
#        
        
#        oval_center24 = origin + np.array([UNIT*2, UNIT * 4])
#        self.oval24 = self.canvas.create_oval(
#            oval_center24[0] - 15, oval_center24[1] - 15,
#            oval_center24[0] + 15, oval_center24[1] + 15,
#            fill='yellow')
#        
#        oval_center25= origin + np.array([UNIT*2, UNIT * 5])
#        self.oval25 = self.canvas.create_oval(
#            oval_center25[0] - 15, oval_center25[1] - 15,
#            oval_center25[0] + 15, oval_center25[1] + 15,
#            fill='yellow')
        
#        oval_center30 = origin + np.array([UNIT*3, UNIT * 0])
#        self.oval30 = self.canvas.create_oval(
#            oval_center30[0] - 15, oval_center30[1] - 15,
#            oval_center30[0] + 15, oval_center30[1] + 15,
#            fill='yellow')
#        
#        oval_center31 = origin + np.array([UNIT*3, UNIT * 1])
#        self.oval31 = self.canvas.create_oval(
#            oval_center31[0] - 15, oval_center31[1] - 15,
#            oval_center31[0] + 15, oval_center31[1] + 15,
#            fill='yellow')
        
        oval_center32 = origin + np.array([UNIT*3, UNIT * 2])
        self.oval32 = self.canvas.create_oval(
            oval_center32[0] - 15, oval_center32[1] - 15,
            oval_center32[0] + 15, oval_center32[1] + 15,
            fill='yellow')
        
#        oval_center33 = origin + np.array([UNIT*3, UNIT * 3])
#        self.oval33 = self.canvas.create_oval(
#            oval_center33[0] - 15, oval_center33[1] - 15,
#            oval_center33[0] + 15, oval_center33[1] + 15,
#            fill='yellow')
        
#        oval_center34 = origin + np.array([UNIT*3, UNIT * 4])
#        self.oval34 = self.canvas.create_oval(
#            oval_center34[0] - 15, oval_center34[1] - 15,
#            oval_center34[0] + 15, oval_center34[1] + 15,
#            fill='yellow')
#        
#        oval_center35 = origin + np.array([UNIT*3, UNIT * 5])
#        self.oval35 = self.canvas.create_oval(
#            oval_center35[0] - 15, oval_center35[1] - 15,
#            oval_center35[0] + 15, oval_center35[1] + 15,
#            fill='yellow')
        
#        oval_center40 = origin + np.array([UNIT*4, UNIT * 0])
#        self.oval40 = self.canvas.create_oval(
#            oval_center40[0] - 15, oval_center40[1] - 15,
#            oval_center40[0] + 15, oval_center40[1] + 15,
#            fill='yellow')
#        
#        oval_center41 = origin + np.array([UNIT*4, UNIT * 1])
#        self.oval41 = self.canvas.create_oval(
#            oval_center41[0] - 15, oval_center41[1] - 15,
#            oval_center41[0] + 15, oval_center41[1] + 15,
#            fill='yellow')
#        
#        oval_center42 = origin + np.array([UNIT*4, UNIT * 2])
#        self.oval42 = self.canvas.create_oval(
#            oval_center42[0] - 15, oval_center42[1] - 15,
#            oval_center42[0] + 15, oval_center42[1] + 15,
#            fill='yellow')
#        
#        oval_center43 = origin + np.array([UNIT*4, UNIT * 3])
#        self.oval43 = self.canvas.create_oval(
#            oval_center43[0] - 15, oval_center43[1] - 15,
#            oval_center43[0] + 15, oval_center43[1] + 15,
#            fill='yellow')
#        
#        oval_center44 = origin + np.array([UNIT*4, UNIT * 4])
#        self.oval44 = self.canvas.create_oval(
#            oval_center44[0] - 15, oval_center44[1] - 15,
#            oval_center44[0] + 15, oval_center44[1] + 15,
#            fill='yellow')
#
#        oval_center50 = origin + np.array([UNIT*5, UNIT * 0])
#        self.oval50 = self.canvas.create_oval(
#            oval_center50[0] - 15, oval_center50[1] - 15,
#            oval_center50[0] + 15, oval_center50[1] + 15,
#            fill='yellow')
        
#        oval_center51 = origin + np.array([UNIT*5, UNIT * 1])
#        self.oval51 = self.canvas.create_oval(
#            oval_center51[0] - 15, oval_center51[1] - 15,
#            oval_center51[0] + 15, oval_center51[1] + 15,
#            fill='yellow')
#        
#        oval_center52 = origin + np.array([UNIT*5, UNIT * 2])
#        self.oval52 = self.canvas.create_oval(
#            oval_center52[0] - 15, oval_center52[1] - 15,
#            oval_center52[0] + 15, oval_center52[1] + 15,
#            fill='yellow')
#        
#        oval_center53 = origin + np.array([UNIT*5, UNIT * 3])
#        self.oval53 = self.canvas.create_oval(
#            oval_center53[0] - 15, oval_center53[1] - 15,
#            oval_center53[0] + 15, oval_center53[1] + 15,
#            fill='yellow')
#        
#        oval_center54 = origin + np.array([UNIT*5, UNIT * 4])
#        self.oval54 = self.canvas.create_oval(
#            oval_center54[0] - 15, oval_center54[1] - 15,
#            oval_center54[0] + 15, oval_center54[1] + 15,
#            fill='yellow')
#        
#        oval_center55 = origin + np.array([UNIT*5, UNIT * 5])
#        self.oval55 = self.canvas.create_oval(
#            oval_center55[0] - 15, oval_center55[1] - 15,
#            oval_center55[0] + 15, oval_center55[1] + 15,
#            fill='yellow')
               
        #pac_center = origin + np.array([UNIT*5, UNIT * 6])
        self.pac = self.canvas.create_arc(
                 origin[0] - 15, origin[1] - 15,
                 origin[0] + 15, origin[1] + 15,
                fill='yellow', start=45, extent=270)

        # pack all
        
        self.canvas.pack()
        ##############        
        #self.bbox = self.canvas.bbox('all')

        
    def reset(self):
        num_flags = 0
        self.update()
        time.sleep(0.5)
        
        
        self.canvas.delete(self.pac)
        
        origin = np.array([20, 20])
        self.pac = self.canvas.create_arc(
                 origin[0] - 15, origin[1] - 15,
                 origin[0] + 15, origin[1] + 15,
                fill='yellow', start=45, extent=270)
        
        self.canvas.delete(self.oval01)
#        self.canvas.delete(self.oval02)
#        self.canvas.delete(self.oval03)
#        self.canvas.delete(self.oval04)
#        self.canvas.delete(self.oval05)
#        self.canvas.delete(self.oval10)
#        self.canvas.delete(self.oval11)
#        self.canvas.delete(self.oval12)
        self.canvas.delete(self.oval13)
#        self.canvas.delete(self.oval14)
#        self.canvas.delete(self.oval15)
#        self.canvas.delete(self.oval20)
#        self.canvas.delete(self.oval21)
#        self.canvas.delete(self.oval23)

#        self.canvas.delete(self.oval24)
#        self.canvas.delete(self.oval25)
#        self.canvas.delete(self.oval30)
#        self.canvas.delete(self.oval31)
        self.canvas.delete(self.oval32)
#        self.canvas.delete(self.oval33)
#        self.canvas.delete(self.oval34)
#        self.canvas.delete(self.oval35)
#        self.canvas.delete(self.oval40)
#        self.canvas.delete(self.oval41)
#        self.canvas.delete(self.oval42)
#        self.canvas.delete(self.oval43)
#        self.canvas.delete(self.oval44)
#        self.canvas.delete(self.oval50)
#        self.canvas.delete(self.oval51)
#        self.canvas.delete(self.oval52)
#        self.canvas.delete(self.oval53)
#        self.canvas.delete(self.oval54)
#        self.canvas.delete(self.oval55)
        
        # return scores_windowreset
        
        return num_flags, self.canvas.coords(self.pac) #, self.canvas.coords(self.hell2), self.canvas.coords(self.hell3)

    def step(self, action, num_flags):
        s = self.canvas.coords(self.pac)
        base_action = np.array([0, 0])
        if action == 0:   # up
            if s[1] > UNIT:
                base_action[1] -= UNIT
        elif action == 1:   # down
            if s[1] < (MAZE_H - 1) * UNIT:
                base_action[1] += UNIT
        elif action == 3:   # right
            if s[0] < (MAZE_W - 1) * UNIT:
                base_action[0] += UNIT
        elif action == 2:   # left
            if s[0] > UNIT:
                base_action[0] -= UNIT
        
        if [s[0]+base_action[0], s[1]+base_action[1], s[2]+base_action[0], s[3]+base_action[1]] in [self.canvas.coords(self.obs1),
           self.canvas.coords(self.obs2)]: #, self.canvas.coords(self.obs3)]:
            s_ = s
            win = 0
            done = False
            reward = -4.0/(6**0.5)
        else:
            self.canvas.move(self.pac, base_action[0], base_action[1])  # move agent
    
            s_ = self.canvas.coords(self.pac)  # next state
    
            # reward function
            if s_ == self.canvas.coords(self.oval01):
                reward = 1/3.0
                num_flags+=1
                done = False
                win = 0
                self.canvas.delete(self.oval01)
#            elif s_ == self.canvas.coords(self.oval02): 
#                reward = 1.0/12.0
#                num_flags+=1
#                done = False
#                self.canvas.delete(self.oval02)
#                #s_ = 'terminal'
#            elif s_ == self.canvas.coords(self.oval03): 
#                reward = 1.0/12.0
#                num_flags+=1
#                done = False
#                self.canvas.delete(self.oval03)
                #s_ = 'terminal'
#            elif s_ == self.canvas.coords(self.oval04): 
#                reward = 1.0/31.0
#                num_flags+=1
#                done = False
#                self.canvas.delete(self.oval04)
#                #s_ = 'terminal'
#            elif s_ == self.canvas.coords(self.oval05): 
#                reward = 1.0/31.0
#                num_flags+=1
#                done = False
#                self.canvas.delete(self.oval05)
#                #s_ = 'terminal'
                
#            elif s_ == self.canvas.coords(self.oval10):
#                reward = 1.0/12.0
#                num_flags+=1
#                done = False
#                self.canvas.delete(self.oval10)
                
#            elif s_ == self.canvas.coords(self.oval11):
#                reward = 1.0/31.0
#                num_flags+=1
#                done = False
#                self.canvas.delete(self.oval11)
#            elif s_ == self.canvas.coords(self.oval12): 
#                reward = 1.0/31.0
#                num_flags+=1
#                done = False
#                self.canvas.delete(self.oval12)
                #s_ = 'terminal'
            elif s_ == self.canvas.coords(self.oval13): 
                reward = 1.0/3.0
                num_flags+=1
                done = False
                win = 0
                self.canvas.delete(self.oval13)
                #s_ = 'terminal'
#            elif s_ == self.canvas.coords(self.oval14): 
#                reward = 1.0/31.0
#                num_flags+=1
#                done = False
#                self.canvas.delete(self.oval14)
#                #s_ = 'terminal'
#            elif s_ == self.canvas.coords(self.oval15): 
#                reward = 1.0/31.0
#                num_flags+=1
#                done = False
#                self.canvas.delete(self.oval15)
                
#            elif s_ == self.canvas.coords(self.oval20):
#                reward = 1.0/12.0
#                num_flags+=1
#                done = False
#                self.canvas.delete(self.oval20)
#            elif s_ == self.canvas.coords(self.oval21):
#                reward = 1.0/12.0
#                num_flags+=1
#                done = False
#                self.canvas.delete(self.oval21)
#            elif s_ == self.canvas.coords(self.oval23):
#                reward = 1.0/12.0
#                num_flags+=1
#                done = False
#                self.canvas.delete(self.oval23)
                
#            elif s_ == self.canvas.coords(self.oval24): 
#                reward = 1.0/31.0
#                num_flags+=1
#                done = False
#                self.canvas.delete(self.oval24)
#                #s_ = 'terminal'
#            elif s_ == self.canvas.coords(self.oval25): 
#                reward = 1.0/31.0
#                num_flags+=1
#                done = False
#                self.canvas.delete(self.oval25)
                
#            elif s_ == self.canvas.coords(self.oval30):
#                reward = 1.0/12.0
#                num_flags+=1
#                done = False
#                self.canvas.delete(self.oval30)
#            elif s_ == self.canvas.coords(self.oval31):
#                reward = 1.0/12.0
#                num_flags+=1
#                done = False
#                self.canvas.delete(self.oval31)
            elif s_ == self.canvas.coords(self.oval32): 
                reward = 1.0/3.0
                num_flags+=1
                done = False
                win = 0
                self.canvas.delete(self.oval32)
                #s_ = 'terminal'
#            elif s_ == self.canvas.coords(self.oval33): 
#                reward = 1.0/12.0
#                num_flags+=1
#                done = False
#                self.canvas.delete(self.oval33)
                #s_ = 'terminal'
#            elif s_ == self.canvas.coords(self.oval34): 
#                reward = 1.0/31.0
#                num_flags+=1
#                done = False
#                self.canvas.delete(self.oval34)
#                #s_ = 'terminal'
#            elif s_ == self.canvas.coords(self.oval35): 
#                reward = 1.0/31.0
#                num_flags+=1
#                done = False
#                self.canvas.delete(self.oval35)
#                
#            elif s_ == self.canvas.coords(self.oval40):
#                reward = 1.0/31.0
#                num_flags+=1
#                done = False
#                self.canvas.delete(self.oval40)
#            elif s_ == self.canvas.coords(self.oval41):
#                reward = 1.0/31.0
#                num_flags+=1
#                done = False
#                self.canvas.delete(self.oval41)
#            elif s_ == self.canvas.coords(self.oval42): 
#                reward = 1.0/31.0
#                num_flags+=1
#                done = False
#                self.canvas.delete(self.oval42)
#                #s_ = 'terminal'
#            elif s_ == self.canvas.coords(self.oval43): 
#                reward = 1.0/31.0
#                num_flags+=1
#                done = False
#                self.canvas.delete(self.oval43)
#                #s_ = 'terminal'
#            elif s_ == self.canvas.coords(self.oval44): 
#                reward = 1.0/31.0
#                num_flags+=1
#                done = False
#                self.canvas.delete(self.oval44)
#                #s_ = 'terminal'
#                
            elif s_ in [self.canvas.coords(self.ghost)]:
                reward = -8.0/(6**0.5)
                done = True
                win = 0
                s_ = 'terminal'
#                
#            elif s_ == self.canvas.coords(self.oval50): 
#                reward = 1.0/31.0
#                num_flags+=1
#                done = False
#                self.canvas.delete(self.oval50)
#                #s_ = 'terminal'    
#            elif s_ == self.canvas.coords(self.oval51): 
#                reward = 1.0/31.0
#                num_flags+=1
#                done = False
#                self.canvas.delete(self.oval51)
#                #s_ = 'terminal'    
#            elif s_ == self.canvas.coords(self.oval52): 
#                reward = 1.0/31.0
#                num_flags+=1
#                done = False
#                self.canvas.delete(self.oval52)
#                #s_ = 'terminal'    
#            elif s_ == self.canvas.coords(self.oval53): 
#                reward = 1.0/31.0
#                num_flags+=1
#                done = False
#                self.canvas.delete(self.oval53)
#                #s_ = 'terminal'    
#            elif s_ == self.canvas.coords(self.oval54): 
#                reward = 1.0/31.0
#                num_flags+=1
#                done = False
#                self.canvas.delete(self.oval54)
#                #s_ = 'terminal'    
#                
#            elif s_ == self.canvas.coords(self.oval55): 
#                reward = 1.0/31.0
#                num_flags+=1
#                done = False
#                self.canvas.delete(self.oval55)
                
        
            else:
                reward = -1.0/15.0
                win = 0
                done = False
                
                
            if num_flags == 3:
                reward = reward+1
                done = True
                win = 1
                s_ = 'terminal'
                
                
        #im = PIL.ImageGrab.grab((0,45,MAZE_W * UNIT, MAZE_H * UNIT+45))
        #im = im.convert('RGB')
        
        #print(im)#.show()
       
        #im.save("/Users/anikamikyan/Desktop/BME595A/Project/THIS_EMAIL/screenshot"+str(num_flags)+".png")

        return s_, reward, done, num_flags, win
        
    def render(self):
        time.sleep(0.1)
        self.update()


def update():
    for t in range(10):
        num_flags, s = env.reset()
        while True:
            env.render()
            a = 1
            s, r, done, num_flags = env.step(a, num_flags)
            if done:
                break

if __name__ == '__main__':
    env = Maze()
    env.after(10, update)
    env.mainloop()
    
    