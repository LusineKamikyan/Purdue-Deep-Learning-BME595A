BME 595 Project (Lusine Kamikyan and Emma Reid)

The project consists of 5 parts:
- main.py
- DQN.py
- maze_env.py
- npmaze.py
- test.py

main.py:
contains the code to to train the agent, moves the agent through the environment using env_maze.py, gets training images from npmaze.py, update the target Q value, and updates the target network, plots the loss, rewards figures

DQN.py
- class Experience Replay which adds an experience to the replay buffer, and pick a batch of experiences for training
- class eps_greedy which implement the epsilon-greedy policy and updates the value of epsilon
- class dqn which is our DQN or Double DQN model depending how we update the target Q value 
- class dueling_dqn which is our Dueling DQN or Dueling Double DQN model depending how we update the target Q value
- env_maze.py contains the environment, the agent, the obstacles, the ghost, the agent moves through the environment and gets rewards, and so on. Anything related o the environment is here
- npmaze.py is the code to obtain equivalent bumpy images to the images from the maze_env.py however, these images make the training much faster
- test.py test the model 