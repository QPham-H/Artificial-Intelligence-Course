import random
import numpy as np

class MDP(): 
	def __init__(self,
			ball_x=None,
			ball_y=None,
			velocity_x=None,
			velocity_y=None,
			paddle_y=None):
		'''
		Setup MDP with the initial values provided.
		'''
		self.create_state(
			ball_x=ball_x,
			ball_y=ball_y,
			velocity_x=velocity_x,
			velocity_y=velocity_y,
			paddle_y=paddle_y
		)

		# the agent can choose between 3 actions - stay, up or down respectively.
		self.actions = [0, 0.04, -0.04]
		#print "Made MDP game"

	def state(self):
		return (self.ball_x, self.ball_y, self.velocity_x, self.velocity_y, self.paddle_y)

	def create_state(self,
			ball_x=None,
			ball_y=None,
			velocity_x=None,
			velocity_y=None,
			paddle_y=None):
		'''
		Helper function for the initializer. Initialize member variables with provided or default values.
		'''
		self.paddle_height = 0.2
		self.ball_x = ball_x if ball_x != None else 0.50
		self.ball_y = ball_y if ball_y != None else 0.50
		self.velocity_x = velocity_x if velocity_x != None else 0.030
		self.velocity_y = velocity_y if velocity_y != None else 0.01
		self.paddle_y = 0.4
		self.reward = 0

	def simulate_one_time_step(self, action_selected):
		'''
		:param action_selected - Current action to execute.
		Perform the action on the current continuous state.
		'''
		# Your Code Goes Here!
		if self.ball_x > 1:
			raise Exception("Ball out of bounds during simulation")
		self.reward = 0
		old_x = self.ball_x
		old_y = self.ball_y

		self.paddle_y = self.paddle_y + action_selected
		if self.paddle_y < 0.:
			self.paddle_y = 0.
		if self.paddle_y > 0.8:
			self.paddle_y = 0.8

		self.ball_y = self.ball_y + self.velocity_y
		if self.ball_y < 0.:
			self.ball_y = -self.ball_y
			self.velcity_y = -self.velocity_y
		if self.ball_y > 1.:
			self.ball_y = 2.-self.ball_y
			self.velocity_y = -self.velocity_y

		self.ball_x = self.ball_x + self.velocity_x
		if self.ball_x < 0.:
			self.ball_x = -self.ball_x
			self.velocity_x = -self.velocity_x
		if self.ball_x > 1.:
			if self.at_paddle(old_x, old_y, self.ball_x, self.ball_y, self.paddle_y):
				self.reward = 1.
				y_vel_flag = True
				while y_vel_flag:
					self.velocity_y = self.velocity_y + random.uniform(-0.03, 0.03)
					if abs(self.velocity_y < 1):
						y_vel_flag = False
				x_vel_flag = True
				while x_vel_flag:
					self.velocity_x = -self.velocity_x + random.uniform(-.015, 0.015)
					if abs(self.velocity_x) > 0.03 and abs(self.velocity_x) < 1:
						x_vel_flag = False
				self.ball_x = 2. - self.ball_x
			else:
#				if self.ball_x > 1:
#					raise Exception("ball_x here is >1")
				self.reward = -1. 


	def at_paddle(self, old_x, old_y, ball_x, ball_y, paddle_y):
		# Need to use slope and delta x since it's continuous time
		ball_y_edge = (ball_y - old_y)/(ball_x - old_x) * (1. - old_x) + old_y
		if old_x > 1.:
			return False
		if ball_y_edge <= paddle_y + 0.2 and ball_y_edge >= paddle_y:
			return True
		else:
			return False

	def discretize_state(self):
		'''
		Convert the current continuous state to a discrete state.
		'''
		# Your Code Goes Here!
		# Return a tuple of discrete states to preserve continuous state values
		if self.ball_x > 1:
			#print "Discretizing a -1"
			return (12, 12, 0, 0, 0)
		if self.velocity_x <= 0:
			dvelocity_x = -1
		else:
			dvelocity_x = 1

		if self.velocity_y < -0.015:
			dvelocity_y = -1
		elif self.velocity_y > 0.015:
			dvelocity_y = 1
		else:
			dvelocity_y = 0

		#if self.ball_x
		dball_x = min(np.floor(12*self.ball_x),11)
		dball_y = min(np.floor(12*self.ball_y),11)
		dpaddle_y = min(np.floor(12*self.paddle_y)/0.8,11)
		return (dball_x, dball_y, dvelocity_x, dvelocity_y, dpaddle_y)

class Simulator:

	def __init__(self, num_games=0, alpha_value=0, gamma_value=0, epsilon_value=0):
		'''
		Setup the Simulator with the provided values.
		:param num_games - number of games to be trained on.
		:param alpha_value - 1/alpha_value is the decay constant.
		:param gamma_value - Discount Factor.
		:param epsilon_value - Probability value for the epsilon-greedy approach.
		'''
		self.num_games = num_games
		self.epsilon_value = epsilon_value
		self.alpha_value = alpha_value
		self.gamma_value = gamma_value
		self.games = 0 #Debugging
	
		# Your Code Goes Here!
		self.Qtable = {}

	def f_function(self, MDP): #gets the best action based on epsilon greedy
		'''
		Choose action based on an epsilon greedy approach
		:return action selected
		'''
		action_selected = None
		if MDP.ball_x > 1:
			#return 0.
			raise Exception("Should've restarted")

		# Your Code Goes Here!
		crazy_prob = random.uniform(0,1)
		#print "This is crazy prob %.2f " %(crazy_prob)
		if crazy_prob < self.epsilon_value:
			action_selected = random.choice(MDP.actions)#How to choose randomly from actions in another object's member
		else:
			biggestQ = -100000.
			for action in MDP.actions:
				list_key = (MDP.discretize_state(), action)
				if list_key not in self.Qtable:
					if biggestQ <= 0:
						action_selected = action
						biggestQ = 0
#					print "This is the action we chose: %d" %(action)
#					return action
				elif self.Qtable[list_key] > biggestQ:
					biggestQ = self.Qtable[list_key]
					action_selected = action
#			if biggestQ == -100000.:
#				action_selected = random.choice(MDP.actions)
		return action_selected

	def train_agent(self):
		'''
		Train the agent over a certain number of games.
		'''
		# Your Code Goes Here!
		total_bounces = 0.
		game_count = 0.
		print "Inside training"
		while game_count < self.num_games:
			total_bounces += self.play_game()
			#print "Just played a game"
			game_count += 1
			self.games += 1 #debugging
			if game_count % 1000 == 0:
				print "Average bounces: %.2f" % (total_bounces/1000)
				print "Game count = "
				print game_count
				total_bounces = 0.
		#return total_bounces

	def play_game(self):
		'''
		Simulate an actual game till the agent loses.
		'''
		# Your Code Goes Here!
		bounce = 0
		game = MDP(0.5,0.5,0.03,0.01,0.4)
#		reward = 0
		#old_reward =0
			#old_reward = reward
#		best_action = self.f_function(game)
#		current_key = (game.discretize_state(), best_action)
#		if current_key not in self.Qtable:
#			self.Qtable[current_key] = self.alpha_value * (reward + self.gamma_value * self.next_Q(game, best_action))
			#print "This is unvisited Qtable value: %.2f" %(self.Qtable[current_key])
			#print "Unvisited of state %s" % (current_key,)
#		else:
#			self.Qtable[current_key] += self.alpha_value * (reward + self.gamma_value * self.next_Q(game, best_action) - self.Qtable[current_key])
		past_reward = 0
		while game.reward >= 0:
			past_reward = game.reward
			best_action = self.f_function(game)
			current_key = (game.discretize_state(), best_action)
			game.simulate_one_time_step(best_action) #game state has updated
			#print "State of the game: "
			#print game.state()
			if game.reward > 0:
				bounce = bounce + 1
#			best_action = self.f_function(game)
#			current_key = (game.discretize_state(), best_action)
			if current_key not in self.Qtable:

				self.Qtable[current_key] = self.alpha_value * (past_reward + self.gamma_value * self.max_Q(game))
#				if self.games % 1 == 0:
#					print "Reward is: %f" % (game.reward)
#				if self.games % 1000 == 0:
#					print "This is unvisited Qtable value: %.2f" %(self.Qtable[current_key])
#					print "Unvisited of state %s" % (current_key,)
			else:
				self.Qtable[current_key] = self.Qtable[current_key] + self.alpha_value * (past_reward + self.gamma_value * self.max_Q(game) - self.Qtable[current_key])
#				if self.games % 1 == 0:
#					print "Reward is: %f" % (game.reward)
#				if self.games % 1000 == 0:
#					print "This is VISITED Qtable value: %.2f" %(self.Qtable[current_key])
#					print "VISITED of state %s" % (current_key,)
			#if bounce == 1:
				#raise Exception ("IT BOUNCED!!")
			#print "Reward is ::: %.2f" %(reward)

#		print "Game lost!!"
		return bounce

	def max_Q (self, MDP):
#		if MDP.ball_x> 1:
#			raise Exception("reward is -1")
#		copy_MDP = copy.deepcopy(MDP)
#		copy_MDP.simulate_one_time_step(best_action)
		max_Q = -100000.
		if MDP.ball_x > 1:
			return -1.
		for action in MDP.actions:
			list_key = (MDP.discretize_state(), action)
			#print "Discretized state of the game with action: "
			#print list_key
			if list_key not in self.Qtable:
				Q = 0
			else:
				Q = self.Qtable[list_key]
			max_Q = max(max_Q, Q)
		return max_Q

def main():
	sim = Simulator(30000, .3, .99, 0.0) #Num of games, alpha, gamma, epsilon
	sim.train_agent()
	bounces = 0.
	n = 0.
	while n < 1000:
		bounces += sim.play_game()
		n += 1
	print "Average number of bounces: "
	print bounces/1000.

main()
