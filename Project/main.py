import matplotlib.pyplot as plt
from maze_env import Maze
import numpy as np
from DQN import eps_greedy
import random
from DQN import Experience_Replay
from DQN import dueling_dqn
from DQN import dqn
import torch
from npmaze_f import MazeArray, show_env
import pickle 

def update():
    # define some parameters we will be using 
    grid_size = 4
    n_episode = 5000
    eps_start = 1.0
    epsilon = 1.0
    eps_end = 0.02
    decay_rate = 0.0001
    
    # discount rate
    gamma = 0.95
    # every N step save the weight from qnetwork to target network
    N_steps = 100
    # batch size from the replay buffer
    batch_replay = 1
    
    batch_size = 5 
    decay_step = 0
    
    for i in range(batch_replay):
        #print(i)
        # We will populate the replay buffer with random experiences then 
        # will use those experiences to train 
        # number of random experiences will be batch size, 
        # then while training we will keep adding to the buffer until max is reached 
        
        # reset the environment
        # get the current state
        num_flags, observation = env.reset()
        
        # putting the flags back into the positions to start the game
        put_goals_back()
        
        #!!!!!!!!!!!
        # Get the image here to give it to e_greedy 
        # this first image will be the initial image  
        # this image is a numpy array of size 6 by 6
        
        
        #########################
        # maze = np.ones((6,6))*.75
        maze = np.ones((grid_size,grid_size))*.75
        ##########################
        maze[0,0] = 1
        maze[1,2] = 0
        maze[2,1] = 0
        maze[0,1] = 0.5
        maze[0,2] = 0.5
        maze[0,3] = 0.5
        maze[1,1] = 0.5
        maze[1,3] = 0.5
        maze[2,0] = 0.5
        maze[2,2] = 0.5
        maze[3,0] = 0.5
        maze[3,2] = 0.5
        # ghost
        maze[3,3] = 0.25
        #maze[2,2] = .25
        
        ############################
#        flags = [(0,1),(0,2),(0,3),(0,4),(0,5),(1,0),(1,1),(1,3),(1,4),(1,5),
#                (2,0),(2,1),(2,3),(2,4),(2,5),(3,0),(3,1),(3,3),(3,4),(3,5),
#                (4,0),(4,1),(4,2),(4,3),(4,4),(4,5), (5,0),(5,1),(5,2),(5,3),(5,5)]
        flags = [(1,0),(2,3),(3,1)]
        #############################
        
        arr = MazeArray(maze,flags)

        np_image = show_env(arr)
        # convert to a torch tensor
        image_tensor = torch.from_numpy(np_image)
        image_tensor = image_tensor.type(torch.FloatTensor)
        

        # if the game is over, reset the game again
        while True:
            # fresh env
            env.render()
            
            # get the random action with probability epsilon, or action that gives
            # max Q training with probability 1-epsilon
            action, eps = e_greedy.update_epsilon(grid_size, epsilon, eps_start, eps_end, image_tensor, decay_rate, decay_step, dqn_train_model)
            #print(action)
            epsilon = eps
            
            # get the next state, reward,... this will be from tkinter
            # the image we get from below line is from the tkinter canvas that we don't use in training
            # we use the np image from npmaze to train the NN
            observation_, reward, done, num_flags, win = env.step(action, num_flags)
            
            
            # to get the next state from np maze
            arr.update_state(action)
            np_image = show_env(arr)
            # convert to a torch tensor
            # next state
            image_tensor_ = torch.from_numpy(np_image)
            image_tensor_ = image_tensor_.type(torch.FloatTensor)

            # save the experience
            # we save the tensor image from npmaze in the experience
            experience_list = replay_buffer.add(observation, action, reward, observation_, done, image_tensor, image_tensor_)
            with open("test.txt", "wb") as fp:  
                pickle.dump(experience_list, fp)
                
        
            # swap the observations
            observation = observation_
            image_tensor = image_tensor_
            
            if done:
                break
       
        
    # optimizer 
    optimizer = torch.optim.Adam(dqn_train_model.parameters(), lr=.0003)
    
    # loss is mean squered error between the Q target value and the Q training value 
    MSE_loss = torch.nn.MSELoss()

    LOSS = np.zeros((n_episode,1))
    EPISODE = np.zeros((n_episode,1))
    REWARD = np.zeros((n_episode,1))
    WIN_NUMBER = np.zeros((n_episode,1))
    
    win_number = 0
    # training
    for episode in range(n_episode):
        Loss = 0
        Reward = 0
        # initial observation
        num_flags, observation = env.reset()
        # putting the goals back in the environment after the previous play
        put_goals_back()
        
        # get the initial step image from npmaze
        maze = np.ones((grid_size,grid_size))*.75
        ##########################
        maze[0,0] = 1
        maze[1,2] = 0
        maze[2,1] = 0
        maze[0,1] = 0.5
        maze[0,2] = 0.5
        maze[0,3] = 0.5
        maze[1,1] = 0.5
        maze[1,3] = 0.5
        maze[2,0] = 0.5
        maze[2,2] = 0.5
        maze[3,0] = 0.5
        maze[3,2] = 0.5
        # ghost
        maze[3,3] = 0.25
        
        #maze[2,2] = .25
        
        ############################
#        flags = [(0,1),(0,2),(0,3),(0,4),(0,5),(1,0),(1,1),(1,3),(1,4),(1,5),
#                (2,0),(2,1),(2,3),(2,4),(2,5),(3,0),(3,1),(3,3),(3,4),(3,5),
#                (4,0),(4,1),(4,2),(4,3),(4,4),(4,5), (5,0),(5,1),(5,2),(5,3),(5,5)]
        flags = [(1,0),(2,3),(3,1)]
        
        arr = MazeArray(maze,flags)
        
        # get the first image
        np_image =  show_env(arr)
        image_tensor = torch.from_numpy(np_image)
        image_tensor = image_tensor.type(torch.FloatTensor)

        average_ind = 0
        while True:
            average_ind = average_ind +1
            # fresh env
            env.render()
        
            
            # get the random action with probability epsilon, or action that gives
            # max Q training with probability 1-epsilon
            action, eps = e_greedy.update_epsilon(grid_size, epsilon,eps_start, eps_end, image_tensor, decay_rate, decay_step, dqn_train_model)

            
            epsilon = eps
            
            # get the next state, reward,... from tkinter
            observation_, reward, done, num_flags, win = env.step(action, num_flags)
            
            # to get the next state from np maze
            arr.update_state(action)
            np_image = show_env(arr)
            # convert to a torch tensor
            # next state
            image_tensor_ = torch.from_numpy(np_image)
            image_tensor_ = image_tensor_.type(torch.FloatTensor)

            # save the experience
            experience_list = replay_buffer.add(observation, action, reward, observation_, done, image_tensor, image_tensor_)
            with open("test.txt", "wb") as fp:  
                pickle.dump(experience_list, fp)
            
            
            Reward = Reward +reward
            # sample a batch of experiences from the replay buffer
            batch_experience = replay_buffer.pick_experience()
            
            
            # get the images,rewards, actions from the batch_experience
            batch_list_im = []
            batch_list_im_current = []
            batch_reward = []
            batch_action = []
            for i in range(batch_size):
                list_reward = torch.Tensor([batch_experience[i][2]])
                list_reward = list_reward.type(torch.FloatTensor)
                list_im_next = batch_experience[i][6]
                list_im_current = batch_experience[i][5]
                list_action = batch_experience[i][1]
                batch_list_im.append(list_im_next)
                batch_list_im_current.append(list_im_current)
                batch_reward.append(list_reward)
                batch_action.append(list_action)
                
                
            stack_next_images = torch.stack(batch_list_im)
            #stack_next_images=stack_next_images.view([1,6,6,10])
            stack_current_images = torch.stack(batch_list_im_current)
            stack_reward = torch.stack(batch_reward)
            # calculate the Q value y = r+gamma*max(Q_target(next state, action))
            
            stack_current_images = stack_current_images.view([batch_size,1,grid_size,grid_size])
            stack_next_images = stack_next_images.view([batch_size,1,grid_size,grid_size])
            
            # Double DQN         
#            ddqn_values = torch.zeros(batch_size,1)
#            for k in range(batch_size):
#                ddqn_values[k] = dqn_target_model.forward(stack_next_images)[k,torch.argmax(dqn_train_model.forward(stack_next_images),1)[k]]
            
            if done == False:
                # DQN
                y = stack_reward + gamma*torch.unsqueeze(torch.max(dqn_target_model.forward(stack_next_images),1)[0],1)
                # Double DQN
                #y = stack_reward+gamma*ddqn_values
            else:
                y = stack_reward
                
            # calculate the loss
            # L = (Q(s,a)-y)^2
            Q_value = torch.zeros([batch_size,1], dtype = torch.float)
            for k in range(len(batch_action)):
                Q_value[k] = dqn_train_model.forward(stack_current_images)[k,batch_action[k]]
                
            # image here will be coming from the sample of
            # experiences from the replay buffer
            y.detach_()
            L = MSE_loss(Q_value,y)
            
            # passing through backward
            L.backward()
            
            # update the parameters
            optimizer.step()
            
            
            # summing up losses to get the loss for each episode
            Loss = Loss+L
            
            
            # swap the observations
            observation = observation_
            image_tensor = image_tensor_
        
            
            # break while loop when end of this episode
            if done:
                break
        
        with open("test.txt", "wb") as fp:  
                pickle.dump(experience_list, fp)
                
        torch.save(dqn_train_model,'/Users/anikamikyan/Desktop/BME595A/LusineKamikyan_BME595_project/dqn_model.pth.tar')

        print('number of flags')
        print(num_flags)
        print('average loss per episode')
        print((1/average_ind)*Loss.detach().numpy())
        print('reward per episode')
        print(Reward)
        LOSS[episode] = (1/average_ind)*Loss.detach().numpy()
        REWARD[episode] = Reward
        EPISODE[episode] = episode
        win_number = win_number+win
        WIN_NUMBER[episode] = win_number
        
        
        print('win per episode')
        print(win)
        # save the target network weights 
        if episode%N_steps == True:
            # save the weight of the training Q network to the target Q network
            for target_param, train_param in zip(dqn_target_model.parameters(), dqn_train_model.parameters()):
                target_param.data.copy_(train_param.data)
                

        decay_step +=1
        
    print('win_number over all episodes')
    print(win_number)
    # plot things
        
    
    plt.figure(1)
    plt.plot(EPISODE, REWARD)
    plt.ylabel('Reward')
    plt.title('DQN Reward')
    plt.show()
    plt.figure(2)
    plt.plot(EPISODE, LOSS)
    plt.ylabel('Average Loss per Episode')
    plt.title('DQN Loss')
    plt.show()
    plt.figure(3)
    plt.plot(EPISODE, WIN_NUMBER)
    plt.ylabel('Win number up to episode')
    plt.title('DQN Win number')
    plt.show()
    # end of game
    print('game over')
    env.destroy()
    return LOSS, EPISODE
    
def put_goals_back():
    origin = np.array([20, 20])
    UNIT = 40

    oval_center01 = origin + np.array([UNIT*0, UNIT * 1])
    env.oval01 = env.canvas.create_oval(
        oval_center01[0] - 15, oval_center01[1] - 15,
        oval_center01[0] + 15, oval_center01[1] + 15,
        fill='yellow')
#    oval_center02 = origin + np.array([UNIT*0, UNIT * 2])
#    env.oval02 = env.canvas.create_oval(
#        oval_center02[0] - 15, oval_center02[1] - 15,
#        oval_center02[0] + 15, oval_center02[1] + 15,
#        fill='yellow')
#    oval_center03 = origin + np.array([UNIT*0, UNIT * 3])
#    env.oval03 = env.canvas.create_oval(
#        oval_center03[0] - 15, oval_center03[1] - 15,
#        oval_center03[0] + 15, oval_center03[1] + 15,
#        fill='yellow')
#    oval_center10 = origin + np.array([UNIT*1, UNIT * 0])
#    env.oval10 = env.canvas.create_oval(
#        oval_center10[0] - 15, oval_center10[1] - 15,
#        oval_center10[0] + 15, oval_center10[1] + 15,
#        fill='yellow')
    oval_center13 = origin + np.array([UNIT*1, UNIT * 3])
    env.oval13 = env.canvas.create_oval(
        oval_center13[0] - 15, oval_center13[1] - 15,
        oval_center13[0] + 15, oval_center13[1] + 15,
        fill='yellow')
#    oval_center20 = origin + np.array([UNIT*2, UNIT * 0])
#    env.oval20 = env.canvas.create_oval(
#        oval_center20[0] - 15, oval_center20[1] - 15,
#        oval_center20[0] + 15, oval_center20[1] + 15,
#        fill='yellow')
#    oval_center21 = origin + np.array([UNIT*2, UNIT * 1])
#    env.oval21 = env.canvas.create_oval(
#        oval_center21[0] - 15, oval_center21[1] - 15,
#        oval_center21[0] + 15, oval_center21[1] + 15,
#        fill='yellow')
#    oval_center23 = origin + np.array([UNIT*2, UNIT * 3])
#    env.oval23  = env.canvas.create_oval(
#        oval_center23[0] - 15, oval_center23[1] - 15,
#        oval_center23[0] + 15, oval_center23[1] + 15,
#        fill='yellow')
#    oval_center30 = origin + np.array([UNIT*3, UNIT * 0])
#    env.oval30 = env.canvas.create_oval(
#        oval_center30[0] - 15, oval_center30[1] - 15,
#        oval_center30[0] + 15, oval_center30[1] + 15,
#        fill='yellow')
#    oval_center31 = origin + np.array([UNIT*3, UNIT * 1])
#    env.oval31 = env.canvas.create_oval(
#        oval_center31[0] - 15, oval_center31[1] - 15,
#        oval_center31[0] + 15, oval_center31[1] + 15,
#        fill='yellow')
    oval_center32 = origin + np.array([UNIT*3, UNIT * 2])
    env.oval32 = env.canvas.create_oval(
        oval_center32[0] - 15, oval_center32[1] - 15,
        oval_center32[0] + 15, oval_center32[1] + 15,
        fill='yellow')
#    oval_center33 = origin + np.array([UNIT*3, UNIT * 3])
#    env.oval33 = env.canvas.create_oval(
#        oval_center33[0] - 15, oval_center33[1] - 15,
#        oval_center33[0] + 15, oval_center33[1] + 15,
#        fill='yellow')
#    

#    oval_center01 = origin + np.array([UNIT*0, UNIT * 1])
#    env.oval01 = env.canvas.create_oval(
#        oval_center01[0] - 15, oval_center01[1] - 15,
#        oval_center01[0] + 15, oval_center01[1] + 15,
#        fill='yellow')
#    
#    oval_center02 = origin + np.array([UNIT*0, UNIT * 2])
#    env.oval02 = env.canvas.create_oval(
#        oval_center02[0] - 15, oval_center02[1] - 15,
#        oval_center02[0] + 15, oval_center02[1] + 15,
#        fill='yellow')
#    oval_center03 = origin + np.array([UNIT*0, UNIT * 3])
#    env.oval03 = env.canvas.create_oval(
#        oval_center03[0] - 15, oval_center03[1] - 15,
#        oval_center03[0] + 15, oval_center03[1] + 15,
#        fill='yellow')
#    oval_center04 = origin + np.array([UNIT*0, UNIT * 4])
#    env.oval04 = env.canvas.create_oval(
#        oval_center04[0] - 15, oval_center04[1] - 15,
#        oval_center04[0] + 15, oval_center04[1] + 15,
#        fill='yellow')
#
#    oval_center05 = origin + np.array([UNIT*0, UNIT *5])
#    env.oval05 = env.canvas.create_oval(
#        oval_center05[0] - 15, oval_center05[1] - 15,
#        oval_center05[0] + 15, oval_center05[1] + 15,
#        fill='yellow')
#    
#    oval_center10 = origin + np.array([UNIT*1, UNIT * 0])
#    env.oval10 = env.canvas.create_oval(
#        oval_center10[0] - 15, oval_center10[1] - 15,
#        oval_center10[0] + 15, oval_center10[1] + 15,
#        fill='yellow')
#    
#    oval_center11 = origin + np.array([UNIT*1, UNIT * 1])
#    env.oval11 = env.canvas.create_oval(
#        oval_center11[0] - 15, oval_center11[1] - 15,
#        oval_center11[0] + 15, oval_center11[1] + 15,
#        fill='yellow')
#    
#    oval_center12 = origin + np.array([UNIT*1, UNIT * 2])
#    env.oval12 = env.canvas.create_oval(
#        oval_center12[0] - 15, oval_center12[1] - 15,
#        oval_center12[0] + 15, oval_center12[1] + 15,
#        fill='yellow')
#    
#    oval_center13 = origin + np.array([UNIT*1, UNIT * 3])
#    env.oval13 = env.canvas.create_oval(
#        oval_center13[0] - 15, oval_center13[1] - 15,
#        oval_center13[0] + 15, oval_center13[1] + 15,
#        fill='yellow')
#    
#    oval_center14 = origin + np.array([UNIT*1, UNIT * 4])
#    env.oval14 = env.canvas.create_oval(
#        oval_center14[0] - 15, oval_center14[1] - 15,
#        oval_center14[0] + 15, oval_center14[1] + 15,
#        fill='yellow')
#    
#    oval_center15 = origin + np.array([UNIT*1, UNIT * 5])
#    env.oval15 = env.canvas.create_oval(
#        oval_center15[0] - 15, oval_center15[1] - 15,
#        oval_center15[0] + 15, oval_center15[1] + 15,
#        fill='yellow')
#    
#    oval_center20 = origin + np.array([UNIT*2, UNIT * 0])
#    env.oval20 = env.canvas.create_oval(
#        oval_center20[0] - 15, oval_center20[1] - 15,
#        oval_center20[0] + 15, oval_center20[1] + 15,
#        fill='yellow')
#    
#    oval_center24 = origin + np.array([UNIT*2, UNIT * 4])
#    env.oval24 = env.canvas.create_oval(
#        oval_center24[0] - 15, oval_center24[1] - 15,
#        oval_center24[0] + 15, oval_center24[1] + 15,
#        fill='yellow')
#    
#    oval_center25= origin + np.array([UNIT*2, UNIT * 5])
#    env.oval25 = env.canvas.create_oval(
#        oval_center25[0] - 15, oval_center25[1] - 15,
#        oval_center25[0] + 15, oval_center25[1] + 15,
#        fill='yellow')
#    
#    oval_center30 = origin + np.array([UNIT*3, UNIT * 0])
#    env.oval30 = env.canvas.create_oval(
#        oval_center30[0] - 15, oval_center30[1] - 15,
#        oval_center30[0] + 15, oval_center30[1] + 15,
#        fill='yellow')
#    
#    oval_center31 = origin + np.array([UNIT*3, UNIT * 1])
#    env.oval31 = env.canvas.create_oval(
#        oval_center31[0] - 15, oval_center31[1] - 15,
#        oval_center31[0] + 15, oval_center31[1] + 15,
#        fill='yellow')
#    
#    oval_center32 = origin + np.array([UNIT*3, UNIT * 2])
#    env.oval32 = env.canvas.create_oval(
#        oval_center32[0] - 15, oval_center32[1] - 15,
#        oval_center32[0] + 15, oval_center32[1] + 15,
#        fill='yellow')
#    
#    oval_center33 = origin + np.array([UNIT*3, UNIT * 3])
#    env.oval33 = env.canvas.create_oval(
#        oval_center33[0] - 15, oval_center33[1] - 15,
#        oval_center33[0] + 15, oval_center33[1] + 15,
#        fill='yellow')
#    
#    oval_center34 = origin + np.array([UNIT*3, UNIT * 4])
#    env.oval34 = env.canvas.create_oval(
#        oval_center34[0] - 15, oval_center34[1] - 15,
#        oval_center34[0] + 15, oval_center34[1] + 15,
#        fill='yellow')
#    
#    oval_center35 = origin + np.array([UNIT*3, UNIT * 5])
#    env.oval35 = env.canvas.create_oval(
#        oval_center35[0] - 15, oval_center35[1] - 15,
#        oval_center35[0] + 15, oval_center35[1] + 15,
#        fill='yellow')
#    
#    oval_center40 = origin + np.array([UNIT*4, UNIT * 0])
#    env.oval40 = env.canvas.create_oval(
#        oval_center40[0] - 15, oval_center40[1] - 15,
#        oval_center40[0] + 15, oval_center40[1] + 15,
#        fill='yellow')
#    
#    oval_center41 = origin + np.array([UNIT*4, UNIT * 1])
#    env.oval41 = env.canvas.create_oval(
#        oval_center41[0] - 15, oval_center41[1] - 15,
#        oval_center41[0] + 15, oval_center41[1] + 15,
#        fill='yellow')
#    
#    oval_center42 = origin + np.array([UNIT*4, UNIT * 2])
#    env.oval42 = env.canvas.create_oval(
#        oval_center42[0] - 15, oval_center42[1] - 15,
#        oval_center42[0] + 15, oval_center42[1] + 15,
#        fill='yellow')
#    
#    oval_center43 = origin + np.array([UNIT*4, UNIT * 3])
#    env.oval43 = env.canvas.create_oval(
#        oval_center43[0] - 15, oval_center43[1] - 15,
#        oval_center43[0] + 15, oval_center43[1] + 15,
#        fill='yellow')
#    
#    oval_center44 = origin + np.array([UNIT*4, UNIT * 4])
#    env.oval44 = env.canvas.create_oval(
#        oval_center44[0] - 15, oval_center44[1] - 15,
#        oval_center44[0] + 15, oval_center44[1] + 15,
#        fill='yellow')
#
#    oval_center50 = origin + np.array([UNIT*5, UNIT * 0])
#    env.oval50 = env.canvas.create_oval(
#        oval_center50[0] - 15, oval_center50[1] - 15,
#        oval_center50[0] + 15, oval_center50[1] + 15,
#        fill='yellow')
#    
#    oval_center51 = origin + np.array([UNIT*5, UNIT * 1])
#    env.oval51 = env.canvas.create_oval(
#        oval_center51[0] - 15, oval_center51[1] - 15,
#        oval_center51[0] + 15, oval_center51[1] + 15,
#        fill='yellow')
#    
#    oval_center52 = origin + np.array([UNIT*5, UNIT * 2])
#    env.oval52 = env.canvas.create_oval(
#        oval_center52[0] - 15, oval_center52[1] - 15,
#        oval_center52[0] + 15, oval_center52[1] + 15,
#        fill='yellow')
#    
#    oval_center53 = origin + np.array([UNIT*5, UNIT * 3])
#    env.oval53 = env.canvas.create_oval(
#        oval_center53[0] - 15, oval_center53[1] - 15,
#        oval_center53[0] + 15, oval_center53[1] + 15,
#        fill='yellow')
#    
#    oval_center54 = origin + np.array([UNIT*5, UNIT * 4])
#    env.oval54 = env.canvas.create_oval(
#        oval_center54[0] - 15, oval_center54[1] - 15,
#        oval_center54[0] + 15, oval_center54[1] + 15,
#        fill='yellow')
#    
#    oval_center55 = origin + np.array([UNIT*5, UNIT * 5])
#    env.oval55 = env.canvas.create_oval(
#        oval_center55[0] - 15, oval_center55[1] - 15,
#        oval_center55[0] + 15, oval_center55[1] + 15,
#        fill='yellow')
#    
    
if __name__ == "__main__":
    env = Maze()
    replay_buffer = Experience_Replay()
    e_greedy = eps_greedy()
    dqn_train_model = dqn()
    dqn_target_model = dqn()
    env.after(10, update)
    env.mainloop()
    
