# based on ~/TUTORIALS/RL
- latest A2C-PPO_BATCH is A2C-PPO_BATCH_.py (works very well)
- lates A2C-PPO_ONLINE is A2C-PPO_ONLINE_2.py (Doesn't really work. Something must be wrong with the implementation)
- make a custom env in Gym for trading
- A2C-PPO-BATHC_.py is about 2x faster with CUDA
- what's the difference between ONLINE and BATCH ? See p.17 http://rail.eecs.berkeley.edu/deeprlcourse-fa17/f17docs/lecture_5_actor_critic_pdf
- see RL_debug.py for some explanations
- why is the critic loss avantage.pow(2).mean() ?
- do you center and normalize rewards in ONLINE?
- action_logprobs explained in A2C-PPO_BATCH.py
- I don't quite understand the ONLINE version of A2C. See p.17 http://rail.eecs.berkeley.edu/deeprlcourse-fa17/f17docs/lecture_5_actor_critic_pdf
- V^pi_theta(s) - is the VALUE of state s _onwards_. This value is NOT the sum of discounted rewards!!! It IS the discouted reward!!! E.g. [0.64 0.81 0.9 1] are V is states s_0, s_1, s_2, s_3. The closer we get the the reward (at the last step in this case) the higher the value.



# david silver course Lecture 7 (DeepMind) -- good
## explains where the log comes from (not just to simplify things computationally)
## Value-based (DQN)
	- the network is a Q(a,s) function. Outputs the value of taking a particular action in state s. In Breakout that would be to predict the score you can get at the end of the game. This is more difficult, than predicting where to move the paddle (up, down). So a policy-based RL algorithm (REINFORCE, A2/3C) may be faster to converge. In a value-based RL algorithm, you select an action based on epsilon-greedy: execute the action that should give the highest reward. (RL Course by David Silver - Lecture 7: Policy Gradient Methods)
	- because of the greediness, it may always behave in the same way in state aliases (not good 18.00)
## Policy-based (A2C, A3C, REINFORCE)
	- the network predicts outputs a distribution of actions, and we sample from that distribution (usually the actions with the highest probability are selected to maximize the reward).
## Optimization
	- Policy can be optimized not only by SGD, other methods exist
	- Gradients can be computed by finite differences (perturb parameters, observe how the function changed, estimate the gradient -- awfully slow, but works with not-differentiable policies 
## Score function
	grad_theta log pi_theta(s,a)
