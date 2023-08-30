
# Import Libraries
import numpy as np
import scipy.io
import math
import matplotlib.pyplot as plt
from scipy import integrate
import pandas as pd
from numpy import savetxt
import time
import csv
import pickle

'''
Intercity Railcar Diesel Mulitple Unit - 5 cars - tractive_eff Characteristics
'''
ICR_size = 5
train_mass = 63 * ICR_size * 1.15 #each car is approx. 63 tonnes
inertia_ratio = 0.08
inertial_train_mass = train_mass * (1 + inertia_ratio)
davis_coef = np.multiply(np.array([6, 0.21, 0.01453, 0]), (404 / 752.4))
engine_kWh = 300000 * ICR_size
train_speed_max = 161
gen_loss = 1.1

# Calculate the maximum tractive force
coef_of_friction = 0.2
gravity = 9.81
powering_axle = (50) * 2
tractive_effort_max = train_mass * 1000 * 9.81 * coef_of_friction * (powering_axle / train_mass) / 1000

# Calculate the maximum acceleration of the train
train_acceleration_max = tractive_effort_max / train_mass
'''
Load the infrastructure characteristcs
'''
# Speed limit profile of the route in kph
route_speed = pd.read_excel(io="route_speed.xlsx")
route_speed = route_speed.to_numpy()

# The terminal station location 
terminus = pd.read_excel(io="terminus.xlsx")
terminus = terminus.to_numpy()

# The stoping pattern of the route
stopping_pattern = pd.read_excel(io="stopping_pattern.xlsx")    # each timing section requires to be inserted with a 1
stopping_pattern = stopping_pattern.to_numpy()                  # if there is a stop the value besdie it is the time planned to be stopped

# The infrastructure gradients along the route
route_gradients = pd.read_excel(io="route_gradients.xlsx")
route_gradients = route_gradients.to_numpy()

# The time to complete each timing section
timing_sections = pd.read_excel(io="A208_timings.xlsx")         # split up evenly from original file
timing_sections = timing_sections.to_numpy()

'''
Define the deceleration rate function for the train 
'''
def deceleration_rate(vel=None, train_acceleration_max=None, engine_kWh=None, inertial_train_mass=None):
        rate = engine_kWh / inertial_train_mass / 1000 / vel
        if rate > train_acceleration_max:
           rate = train_acceleration_max
        rate = 0.5

        return rate

'''
Definition of the step size in metres for the train calculation 
'''
Step_min = 0
Step_max = np.amax(route_speed[:, 0]) * 1000
Step_size = 1

route_lenght = np.arange(Step_min, (Step_max + 2), Step_size)
route_dist = len(route_lenght)

'''
Create arrays to store the calculations as the train progresses from each step size 
'''
section_Time = np.zeros((route_dist,))
forward_vel = np.zeros((route_dist,))
deceleration = np.zeros((route_dist,))
train_velocity = np.zeros((route_dist,))
time_delta = np.zeros((route_dist,))
run_time = np.zeros((route_dist,))
fwd_tract_delta = np.zeros((route_dist,))
route_velocity_limit = np.zeros((route_dist,))
line_train_velocity_limit = np.zeros((route_dist,))
curr_gradient_profile = np.zeros((route_dist,))
tractive_eff_fwd = np.zeros((route_dist,))
tractive_eff_bkwd = np.zeros((route_dist,))
engine_kWh_output = np.zeros((route_dist,))
fuel_consumption = np.zeros((route_dist,))
actutal_accel = np.zeros((route_dist,))
fwd_accel = np.zeros((route_dist,))
bkwd_accel = np.zeros((route_dist,))
fwd_tract = np.zeros((route_dist,))
tractive_eff = np.zeros((route_dist,))
train_velocity = np.zeros((route_dist,))
power_notch_applied = np.zeros((route_dist,))

'''
Calculate the actual velocity profile in m/s
'''
curr_position = 1
# Iterate over the route length to calculate the train velocity limits
for i in np.arange(0, route_dist-1):
    route_velocity_limit[i] = route_speed[(curr_position-1), 1] / 3.6
    line_train_velocity_limit[i] = route_velocity_limit[i]
    if route_velocity_limit[i] >= train_speed_max / 3.6:
        route_velocity_limit[i] = train_speed_max / 3.6
    if i * Step_size >= np.fix(route_speed[curr_position, 0] * 1000):
        curr_position += 1

'''
Calculate the acutal gradient for the given location
'''
curr_position = 1
for i in np.arange(1,(math.floor(1000 / Step_size * np.amax(route_gradients[:,0])))):
  curr_gradient_profile[i] = route_gradients[(curr_position - 1),1] / 1000 * gravity / (1 + inertia_ratio)
  if i * Step_size >= np.trunc(route_gradients[curr_position,0] * 1000):
    curr_position += 1

curr_gradient_profile[i + 1] = curr_gradient_profile[i]

'''
Calculate the terminus,station stopping pattern and station dwell time (if applicable)
'''
complete = 0
t_point = 0
s_point = stopping_pattern.shape
s_point_1 = (1,1)
s_point = np.subtract(s_point, s_point_1)
journey = [0]
station_dwell_time = [0]
# Check the 
if stopping_pattern[t_point, 0] != 0:
    n = -1
else:
    n = 0
# Loop through the route to update the journey profile
while complete != 1:
    t_point = t_point + 1
    # Check to see if the stop is the terminus
    if stopping_pattern[t_point, 0] == 0:
        complete = 1
        t_point_maximum = t_point - 1
    else:
        # Extract the journey profile and dwell times 
        if t_point == s_point[0]:
            complete = 1
            journey.insert(t_point, ((np.fix(stopping_pattern[t_point + n, 0] / Step_size)) * Step_size))
            station_dwell_time.insert(t_point, (stopping_pattern[t_point + n, 1]))
            if stopping_pattern[t_point, 0] == 0:
                k = [(s_point) / Step_size for s_point in journey]
                for i in k:
                    route_velocity_limit[int(i)] = 0
            t_point_maximum = t_point
        else:
            journey.insert(t_point, (np.trunc(stopping_pattern[(t_point + n), 0] / Step_size)) * Step_size)
            station_dwell_time.insert(t_point, stopping_pattern[t_point + n, 1])
            if stopping_pattern[t_point, 0] == 0:
                k = [(s_point) / Step_size for s_point in journey]
                for i in k:                    
                    route_velocity_limit[int(i)] = 0
# Ensure that the final stop is include in the journey profile
if journey[t_point_maximum-1] != (np.trunc(Step_max / Step_size)) * Step_size:
    t_point_maximum = t_point_maximum + 1
    journey.insert((t_point_maximum), (np.trunc(Step_max / Step_size) * Step_size))
# Set the terminus to calculate dwell time as zero
station_dwell_time[t_point_maximum-1] = 0

'''
Create the State for Q-Learning
Citation:   Q-Learning Agent base code used from MARL Module 2023
'''
class State:
    def __init__(self, state=i):
        self.state = state
        self.isEnd = False
        self.PUNISH = False
        self.REWARD = False

    # Define reward function
    def getReward(self, punish = 1, reward = 1):
        if punish == 1:
            return time_pun
        elif punish == 10:
            return fuel_pun
        elif reward == 3:
            return time_rew
        elif reward == 10:
            return fuel_rew
        else:
            return 0
    
    # Define state end function
    def isEndFunc(self, punish = 1):
        if (self.state == route_dist-1) or (punish == 1):
            self.isEnd = True
    
    # Define the agents next action
    def nxtPosition(self, action):
        if action == 0.0001:
            selected_power_notch = 0.0001
        elif action == 0.2:
            selected_power_notch = 0.2
        elif action == 0.4:
            selected_power_notch = 0.4
        elif action == 0.6:
            selected_power_notch = 0.6
        elif action == 0.8:
            selected_power_notch = 0.8
        else:
            selected_power_notch = 1
        return selected_power_notch

class Agent:
    def __init__(self):
        self.reward_per = np.array([]) # array to coollect the agent rewards per episode
        '''
        Actions are the power notch control available to the agent
        Two arrays for tests a & b
        '''
        if power_notch_control == "all":
            self.actions = [0.0001, 0.2, 0.4, 0.6, 0.8, 1] # N, 1, 2, 3, 4, 5
        elif power_notch_control == "max":
            self.actions = [0.0001, 1] # N, Max_Power ##### Howlett's Power-Coast
        
        self.State = State()
        self.discount = DISCOUNT
        self.lr = LEARNING_RATE
        self.PUNISH = 0
        self.REWARD = 0
        self.action_dist_limit = 250 # limit the agents choice of actions to every n metres (250 works best)
        self.time_section_epsilon = {}  # Timing section epsilon dictionary
        self.fuel_per_episode = {} # dictionary to collect the fuel per episode
        self.fuel_first_pass = {} # dictionary to collect the fuel from teh first successful pass
        self.beta = {} # dictionary for beta value for fuel reduction
        self.fuel_qlearner = {} #dictionary to collect the fuel consumed by the agent

        # Create the dictionaries
        for t_point in range(t_point_maximum):
            self.time_section_epsilon[(t_point)] = EPSILON
            self.beta[(t_point)] = 0
            self.fuel_per_episode[(t_point)] = 0
            self.fuel_first_pass['TIP_' + str(t_point)] = np.array([])
            self.fuel_qlearner['TIP_' + str(t_point)] = np.array([])

        self.action_values = {} # dictionary for the agents actions
        self.delta = {} # dictionary for action delta

        for i in range(route_dist//(self.action_dist_limit)+1):
            for notch in self.actions:
                self.action_values[(i, notch)] = 0
                self.delta[(i, notch)] = 0    
    
    # Define agent action choice
    def chooseAction(self, curr_position, t_point):
        # choose action according to policy eps-greedy
        if np.random.uniform(0, 1) <= self.time_section_epsilon[(t_point)]:
            action = np.random.choice(self.actions)
        else:
            action = self.best_action(curr_position)
        return action
    
    # Define agents action
    def takeAction(self, action):
            notch_position = self.State.nxtPosition(action)
            self.State.state = notch_position
    
    # Define the agents best action
    def best_action(self, curr_position):
        best = -1
        max_val = -100000000
        for notch in self.actions:
            q_val = self.action_values[curr_position, notch]
            if q_val >= max_val:
                max_val = q_val
                best = notch
        return best
    
    # Define the Q Max Value for the agent
    def q_max(self, curr_position):
        best = self.best_action(curr_position)
        return self.action_values[curr_position, best]
    
    # Define the Q-Learning Function for the agent
    def q_learning(self, episodes):   
        # Set out the initial variables
        x, printcounter, action_dist_counter, finished = 0, 0, 0, 0
        self.State.state = list(range(0,route_dist))
        beta = 0.1
        self.agent_section_times = np.array([]) # create an array for the agent times for retrospective analysis    
        start = time.time() # for strat finish time analysis to run the code
        
        # Start the iteration through each episode
        while x < episodes:
        
            fuel_episode_total = 0
            action_dist_counter = 0
            action_counter = []
            self.action_list = []
            action_prev = []
            counter = 0
            self.State.isEnd = False
            self.epsiode_time = np.array([])
            episode_fuel_consumption = []
            self.epsiode_time = np.append(self.epsiode_time, [123456789, x/1000]) # collect times for agent 
            agent_reward = np.array([]) # Collect the agent rewards

            '''
            Start the iteration through each timing section
            '''
            for t_point in np.arange(0,(t_point_maximum),1):
                # Set out the start and end points for the current timing section
                t_point_start, t_point_end = (journey[(t_point)]), (journey[(t_point+1)])
                timing_section = np.arange(math.floor(t_point_start/(Step_size )),math.floor(t_point_end/(Step_size)+1))
                # Collect the actions for the current section
                action_pool = []
                
                '''
                Calculate the forward velocity of the train
                '''          
                for i in timing_section:
                    # Calculate the rolling resistance
                    rolling_resistance = [(davis_coef[0,] + davis_coef[1,] * forward_vel[i] + davis_coef[2,] * (forward_vel[i]) ** 2) / inertial_train_mass]
                    # Calculate the forward traction delta
                    fwd_tract_delta[(i + 1)] = 2 * Step_size / (forward_vel[i] + forward_vel[(i + 1)])
                    # Calculate the foward traction
                    if forward_vel[i] == 0:
                        fwd_tract[(i + 1)] = (fwd_tract[i] + 1)
                    else:
                        fwd_tract[(i + 1)] = fwd_tract[i] + fwd_tract_delta[(i + 1)]

                    # Set the power notch aplied for the next n metres (defined in action_dist_limit
                    if action_dist_counter >0 and action_dist_counter <= self.action_dist_limit:
                        power_notch_applied[i] = power_notch_applied[i-1]
                        if action_dist_counter == self.action_dist_limit-1:
                            # Reset to zero
                            action_dist_counter = 0
                        else:
                            # Add one metre
                            action_dist_counter += 1
                    else:
                        # Store the previous action
                        previous_power_notch = power_notch_applied[i]
                        # Calculate the current speed in kph
                        speed = forward_vel[i]*3.6//1
                        # Calculate the current speed limit in kph
                        speed_lim = route_velocity_limit[i]*3.6//1
                        # Ensure power notch is not equal to zero to avoid divide by zero errors 
                        if previous_power_notch == 0:
                          previous_power_notch = 0.0001
                        # Agent current state
                        currentState = self.State.state[counter]
                        if counter < route_dist//(self.action_dist_limit):
                          counter +=1
                        # Ensure the agent does not exceed the speed limit
                        if (speed+3) >= speed_lim:
                            # Apply coast
                            power_notch_applied[i] = 0.0001
                        else:
                            # Agent chooses next action!
                            power_notch_applied[i] = self.chooseAction(currentState, t_point)
                        # Append the location of the action taken                        
                        action_counter.append(i) # This is required for reuse in the time section 
                        # Append the action taken
                        self.action_list.append(power_notch_applied[i])
                        # Appedn the previous action
                        action_prev.append(previous_power_notch)
                        # Add one metre 
                        action_dist_counter +=1

                    # Calculate the tractive effort applied
                    tractive_eff[i] = engine_kWh * (power_notch_applied[i] / (forward_vel[i]) / (inertial_train_mass * 1000))
                    # Ensure that the tractive effort calculation does not exceed the max accelaration available to the train
                    if tractive_eff[i] > train_acceleration_max:
                        tractive_eff[i] = train_acceleration_max
                    # Calculate the forward accelaration
                    fwd_accel[i + 1] = tractive_eff[i] - rolling_resistance - curr_gradient_profile[i]
                    # Ensure that the velocity does not exceed the limit 
                    if forward_vel[i] < route_velocity_limit[i + 1]:
                        forward_vel[i + 1] = (forward_vel[i] ** 2 + 2 * fwd_accel[i + 1] * (Step_size)) ** 0.5
                        # Reset the speed to the limit if exceed - this is not ideal
                        if forward_vel[i + 1] > route_velocity_limit[i + 1]:
                            forward_vel[i + 1] = route_velocity_limit[i + 1]
                    else:
                        # Ensure that the forward velocity has not calculated a negative value
                        if (fwd_accel[i + 1] < 0):
                            forward_vel[i + 1] = (forward_vel[i] ** 2 + 2 * (fwd_accel[i + 1]) * (Step_size)) ** 0.5
                            # Ensure that the velocity does not exceed the limit 
                            if forward_vel[i + 1] > route_velocity_limit[i + 1]:
                                # Reset the speed to the limit if exceed - this is not ideal
                                forward_vel[i + 1] = route_velocity_limit[i + 1]
                        else:
                            # Set to accelaration to zero 
                            fwd_accel[i + 1] = 0
                            # Ensure the current speed limit is not equal to zero
                            if route_velocity_limit[i + 1] == 0:
                                forward_vel[i + 1] = 0
                            else:
                                forward_vel[i + 1] = route_velocity_limit[i]

                '''
                Calculate the deceleration rate of the train to ensure that the train stops at the required location
                Not optimal, this loops though the timing section in reverse to cacluate the backward velocity
                '''
                i = math.floor(t_point_end / Step_size)
                # Calculate the rolling resistance
                rolling_resistance = (davis_coef[0,] + davis_coef[1,] * deceleration[i+1] + davis_coef[2,] * (deceleration[i+1]) ** 2) / inertial_train_mass
                # Calculate the backward acceleration
                bkwd_accel[i] = rolling_resistance + deceleration_rate(deceleration[i+1],train_acceleration_max,engine_kWh,inertial_train_mass) + curr_gradient_profile[i+1]
                # The decelaration velocity
                deceleration[i] = (deceleration[i+1] ** 2 + 2 * bkwd_accel[i] * (Step_size)) ** 0.5
                # Loop through the timing section in reverse
                for i in np.arange(math.floor(t_point_end-1 / Step_size),math.floor(t_point_start / Step_size),- 1):
                    # If the deceleration velocity is less than the next metres current speed limit caculate the rates
                    if deceleration[i+1] < route_velocity_limit[i+1]:
                        # Calculate the rolling resistance
                        rolling_resistance = (davis_coef[0,] + davis_coef[1,] * deceleration[i+1] + davis_coef[2,] * (deceleration[i+1]) ** 2) / inertial_train_mass
                        # Calculate the backward acceleration
                        bkwd_accel[i] = rolling_resistance + deceleration_rate(deceleration[i+1],train_acceleration_max,engine_kWh,inertial_train_mass) + curr_gradient_profile[i+1]
                        # The decelaration velocity
                        deceleration[i] = (deceleration[i+1] ** 2 + 2 * bkwd_accel[i] * (Step_size)) ** 0.5
                    # Ensuring that the train decelerates one time step ahead of the limit
                    elif route_velocity_limit[i+1] < route_velocity_limit[i]:
                        # Calculate the rolling resistance
                        rolling_resistance = (davis_coef[0,] + davis_coef[1,] * deceleration[i+1] + davis_coef[2,] * (deceleration[i+1]) ** 2) / inertial_train_mass
                        # Calculate the backward acceleration
                        bkwd_accel[i] = rolling_resistance + deceleration_rate(deceleration[i+1],train_acceleration_max,engine_kWh,inertial_train_mass) + curr_gradient_profile[i+1]
                        # The decelaration velocity
                        deceleration[i] = (deceleration[i+1] ** 2 + 2 * bkwd_accel[i] * (Step_size)) ** 0.5
                    else:
                        # The decelaration velocity to current speed limit
                        deceleration[i] = route_velocity_limit[i]               
                '''
                Next to combine the forward and backward velocities to get an actaul speed profile for the train
                '''
                # Set up the fuel for the current timing section
                fuel_in_timing_section = 0
                timing_section_vel = np.arange(math.floor(t_point_start / Step_size),math.floor(t_point_end / Step_size))
                # Loop through the section again
                for i in timing_section_vel:
                    # Check velocity profiles for the current metre and update the actual velocity 
                    if forward_vel[i] <= deceleration[i]:
                        actutal_accel[i] = fwd_accel[i]
                        train_velocity[i] = forward_vel[i]
                    else:
                        actutal_accel[i] = - bkwd_accel[i]
                        train_velocity[i] = deceleration[i]
                    
                    # Calculate the actual rolling resistance now the velocity profile has been updated for forward and backward
                    rolling_resistance = (davis_coef[0,] + davis_coef[1,] * train_velocity[i] + davis_coef[2,] * (train_velocity[i]) ** 2) / inertial_train_mass
                    # Calculate the engine kilo Watt hours output
                    engine_kWh_output[i] = (actutal_accel[i] + rolling_resistance + curr_gradient_profile[i]) * inertial_train_mass * 1000
                    # Calculate the tractive effort used given only that traction is being applied
                    if engine_kWh_output[i] > 0:
                        tractive_eff_bkwd[i] = 0
                        if engine_kWh_output[i] > tractive_effort_max * 1000:
                            tractive_eff_fwd[i] = tractive_effort_max * 1000
                        else:
                            tractive_eff_fwd[i] = engine_kWh_output[i]
                    else:
                        tractive_eff_fwd[i] = 0
                        tractive_eff_bkwd[i] = - engine_kWh_output[i]
                    # Calculate the actual fuel consumption
                    fuel_consumption[i] = tractive_eff_fwd[i]/1000000/10.96/gen_loss
                    # Update the fuel consumed through the section
                    fuel_in_timing_section += fuel_consumption[i]

                '''
                Next to calculate the time throughout each section
                '''
                i = math.floor(t_point_start / Step_size+1)
                # Calculate the station dwell time (if in use)
                time_delta[i + 1] = 2 * Step_size / (train_velocity[i] + train_velocity[i + 1]) + station_dwell_time[t_point]
                # Update the section running time
                section_Time[i + 1] = section_Time[i] + time_delta[i + 1]
                # appende the fuel consumed with the current timing section 
                episode_fuel_consumption.append(round(fuel_in_timing_section, 2))
                # Update the total fuel consumed for the episode
                fuel_episode_total += fuel_in_timing_section
                # Round down the value
                fuel_episode_total = round(fuel_episode_total,1)
                # Loop through the time section to cacluate the time
                for i in np.arange(math.floor(t_point_start / Step_size+1), math.floor(t_point_end / Step_size+1)):
                    # Calculate the time taken between metres
                    time_delta[i + 1] = 2 * Step_size / (train_velocity[i] + train_velocity[i + 1])
                    # For the action reward it is required to load the stored action taken at this location
                    if i in action_counter:
                        # Reset the current state for the agent
                        currentState = self.State.state[i]
                        # Store the nth action
                        j = i//self.action_dist_limit
                        # Append the action number for the reward function
                        action_pool.append(j)                     
                    # First two episodes this is required to allow the time calculation return no inf values
                    if x <= 1:
                        # update the runnning time
                        run_time[i] = run_time[i-1] + time_delta[i]
                    # Now to calculate the run time for the agent and rewards
                    else:
                        # Actual run time
                        run_time[i] = run_time[i-1] + time_delta[i]
                        # Remove milliseconds
                        Time =  run_time[i]//1
                        # Update the fuel per episode
                        self.fuel_per_episode[(t_point)] = fuel_in_timing_section
                        # Iterate through the timing sections to check the scheduled time
                        for k in range(len(timing_sections)):
                            # If the current location is the timing point check the time
                            if i == timing_sections[k, 0]:
                                # Verifying the time for the reward function with a 30 sec tolerance
                                if Time <= timing_sections[k, 1]+time_tol and Time >= timing_sections[k, 1]-time_tol:
                                    # Give time reward
                                    self.REWARD = 3
                                    # Append the details of the time to complete (for analysis and debug)
                                    self.epsiode_time = np.append(self.epsiode_time, k/10)
                                    self.epsiode_time = np.append(self.epsiode_time, Time)
                                    self.epsiode_time = np.append(self.epsiode_time, timing_sections[k, 1])
                                    # Break out to not check the fuel consumption reward 
                                    if finished < 1:
                                        break
                                    # Now to reward the fuel consumed
                                    else:
                                        # Load the fuel consumption for this timing section from the first pass minus beta
                                        first_pass_fuel = np.mean(self.fuel_first_pass['TIP_' + str(t_point)])-self.beta[t_point]
                                        # If equal reward the correct time reward
                                        if round(first_pass_fuel,2) == round(episode_fuel_consumption[t_point],2):                              
                                            self.REWARD = 3
                                      # If more fuel efficient give fuel reward
                                        elif round(first_pass_fuel,2) > round(episode_fuel_consumption[t_point],2):                                         
                                            self.REWARD = 10
                                            # Update beta value
                                            self.beta[t_point] += 0.05
                                            # Round to ensure values remian two decimal places
                                            self.beta[t_point] = round(self.beta[t_point],2)
                                            # Set upper limit for beta
                                            if self.beta[t_point] >1.2:
                                                self.beta[t_point] = 1.2
                                        # If not more fuel efficient reward lower positive value
                                        else:
                                            self.PUNISH = 10
                                            # Update beta value
                                            self.beta[t_point] -= 0.01
                                            # Round to ensure values remian two decimal places
                                            self.beta[t_point] = round(self.beta[t_point],2)
                                            # Set lower limit for beta 
                                            if self.beta[t_point] <-0.9:
                                                self.beta[t_point] = -0.9
                                            break
                                    break
                                # Punish the agent for not adhereing to the time
                                else:
                                    # Give punishment
                                    self.PUNISH = 1
                                    # Append the details of the time to complete (for analysis and debug)
                                    self.epsiode_time = np.append(self.epsiode_time, k/10)
                                    self.epsiode_time = np.append(self.epsiode_time, Time)
                                    self.epsiode_time = np.append(self.epsiode_time, timing_sections[k, 1])
                                    break                       
                        # Loop thorugh rewrad function only if rewards were given
                        if self.REWARD == 3 or self.REWARD == 10 or self.PUNISH == 1:
                            # get reward
                            reward = self.State.getReward(self.PUNISH, self.REWARD)                          
                            agent_reward = np.append(agent_reward, reward) # collect the reward after each action
                            # Check if is end for agent
                            self.State.isEndFunc(self.PUNISH)
                            # Reset punish and reward to zeros
                            self.PUNISH, self.REWARD = 0, 0                            
                            # Loop through the actions taken to assign the avegered reward
                            for curr_position in action_pool:
                                # Reload saved action for current position
                                notch = self.action_list[curr_position]
                                # Reload saved action for previous position
                                notch_prev = action_prev[curr_position]
                                # Load old q
                                old_q = self.action_values[curr_position, notch]
                                # Determine next state                                
                                next_state = self.State.state[curr_position+1]
                                # Get max Q
                                max_q = self.q_max(next_state)
                                # Calculate the average of teh reward for the number of actions in the timing section
                                rew_avg = reward/len(action_pool)                                
                                if reward > 0:
                                    # Update delta for being a postive reward
                                    self.delta[curr_position, notch] += 0.1
                                    # Decay epsilon for the timing section
                                    self.time_section_epsilon[(t_point)] *= 0.9985
                                    # Set upper limit on delta
                                    if self.delta[curr_position, notch] >1:
                                        self.delta[curr_position, notch] = 1                                
                                else:
                                    # Update delta for being a postive reward    
                                    self.delta[curr_position, notch] -= 0.01
                                    # Set lower limit on delta
                                    if self.delta[curr_position, notch] <-1:                                        
                                        self.delta[curr_position, notch] = -1
                                # Update current delta value
                                t_delta = self.delta[curr_position, notch]
                                # Calculate the new Q value
                                new_q = old_q + self.lr * (rew_avg + t_delta + self.discount * max_q - old_q)
                                # Update the Q-Table
                                self.action_values[curr_position, notch] = new_q
                                # If the agenhas successfully completed the journey
                                if curr_position == 325 and reward > 5: ######## the value 325 will need to be updated depending the journey size ######
                                    # Print the various details to monitor progress
                                    print("epoch = ", run_in, "run = ", x, "FINISH!!!", finished, agent_reward)
                                    print(fuel_episode_total)
                                    # Update finished to ensure Q-Learning can occur
                                    finished += 1
                                    # Loop through to capture fuel consumption
                                    for i in range(len(episode_fuel_consumption)):
                                        # First pass
                                        if finished < 2:
                                            self.fuel_first_pass['TIP_' + str(i)] = np.append(self.fuel_first_pass['TIP_' + str(i)], round(episode_fuel_consumption[i],2))
                                        # Agents fuel consumption after rewards are enabled
                                        else:
                                            self.fuel_qlearner['TIP_' + str(i)] = np.append(self.fuel_qlearner['TIP_' + str(i)], round(episode_fuel_consumption[i],2))
                            # Check if state is end to break loop
                            if self.State.isEnd == True:
                                break
                        # Append reward
                        reward_sum = np.sum(agent_reward) # sum the rewards for the episode
                # Check if state is end to break loop
                if self.State.isEnd == True:
                    break
            # Calculate reward sum per episode
            reward_sum = np.sum(agent_reward) # sum the rewards for the episode
            # Append rewards from each episode
            self.reward_per = np.append(self.reward_per, reward_sum) # Store the total reward for each episode
            self.agent_section_times = np.append(self.agent_section_times, self.epsiode_time, axis = 0) 
            # Check episode number to store files locally for analysis
            if (printcounter == 100):
                # Get program run end time
                end = time.time()
                print('Progress report...', x, run_time.max())
                print(end - start)
                # Save out files for analysis
                savetxt(f'timing{x}{run_in}.csv', self.agent_section_times, delimiter=',')
                savetxt(f'rewards{x}{run_in}.csv', self.reward_per, delimiter=',')
                
                '''
                Save out various dictionaries, variables and for analysis through each epoch
                '''   
                with open(f'fuel_first_pass_epoch-{run_in}.pkl', 'wb') as fp:
                    pickle.dump(ag.fuel_first_pass, fp)

                with open(f'delta_epoch-{run_in}.pkl', 'wb') as fp:
                    pickle.dump(ag.delta, fp)

                with open(f'action_values_epoch-{run_in}.pkl', 'wb') as fp:
                    pickle.dump(ag.action_values, fp)
         
                with open(f"fuel_first_pass_epoch-{run_in}.csv", "w", newline="") as fp:
                    writer = csv.DictWriter(fp, fieldnames=ag.fuel_first_pass.keys())
                    writer.writeheader()
                    writer.writerow(ag.fuel_first_pass)

                with open(f"beta_dict_epoch-{run_in}.csv", "w", newline="") as fp:
                    writer = csv.DictWriter(fp, fieldnames=ag.beta.keys())
                    writer.writeheader()
                    writer.writerow(ag.beta)

                with open(f"fuel_q-learner_epoch-{run_in}.csv", "w", newline="") as fp:
                    writer = csv.DictWriter(fp, fieldnames=ag.fuel_qlearner.keys())
                    writer.writeheader()
                    writer.writerow(ag.fuel_qlearner)

                with open(f"train_velocity_epoch-{run_in}.csv", "w", newline="") as fp:
                    writer = csv.writer(fp, delimiter=' ')
                    writer.writerow(train_velocity)

                with open(f"agent_action_list_epoch-{run_in}.csv", "w", newline="") as fp:
                    writer = csv.writer(fp, delimiter=' ')
                    writer.writerow(ag.action_list)
                # Restart timer for each epoch
                start = time.time()
                printcounter = 0

            printcounter += 1
            x += 1

if __name__ == "__main__":
    # Set Hyperparameters
    DISCOUNT = 0.90
    LEARNING_RATE = 0.50
    EPSILON = 0.10
    '''
        "all" is all 5 power notch controls available and neutral
        "max" is all maximum power notch control and neutral only
    '''
    power_notch_control = "all" # Please set to which method you would like to test
    # Enable n epochs of Q-Learning
    for i in range(10):
        ag = Agent()
        # Set time tolerance
        time_tol = 27
        # Set Rewards
        fuel_rew = 15
        time_rew = 10
        time_pun = -1
        fuel_pun = 8
        run_in = i
        # Enable learing for n episodes
        ag.q_learning(501)



