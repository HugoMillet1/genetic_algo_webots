from controller import Robot, Receiver, Emitter
import sys,struct,math
import numpy as np
import mlp as ntw

class Controller:
    def __init__(self, robot):        
        # Robot Parameters
        # Please, do not change these parameters
        self.robot = robot
        self.time_step = 32 # ms
        self.max_speed = 1  # m/s
 
        # MLP Parameters and Variables   
        ### Define below the architecture of your MLP network. 
        ### Add the number of neurons for each layer.
        ### The number of neurons should be in between of 1 to 20.
        ### Number of hidden layers should be one or two.
        self.number_input_layer = 11
        # Example with one hidden layers: self.number_hidden_layer = [5]
        # Example with two hidden layers: self.number_hidden_layer = [7,5]
        self.number_hidden_layer = [16, 16] 
        self.number_output_layer = 2
        
        # Create a list with the number of neurons per layer
        self.number_neuros_per_layer = []
        self.number_neuros_per_layer.append(self.number_input_layer)
        self.number_neuros_per_layer.extend(self.number_hidden_layer)
        self.number_neuros_per_layer.append(self.number_output_layer)
        
        # Initialize the network
        self.network = ntw.MLP(self.number_neuros_per_layer)
        self.inputs = []
        
        # Calculate the number of weights of your MLP
        self.number_weights = 0
        for n in range(1,len(self.number_neuros_per_layer)):
            if(n == 1):
                # Input + bias
                self.number_weights += (self.number_neuros_per_layer[n-1]+1)*self.number_neuros_per_layer[n]
            else:
                self.number_weights += self.number_neuros_per_layer[n-1]*self.number_neuros_per_layer[n]

        # Enable Motors
        self.left_motor = self.robot.getDevice('left wheel motor')
        self.right_motor = self.robot.getDevice('right wheel motor')
        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)
        self.velocity_left = 0
        self.velocity_right = 0
    
        # Enable Proximity Sensors
        self.proximity_sensors = []
        for i in range(8):
            sensor_name = 'ps' + str(i)
            self.proximity_sensors.append(self.robot.getDevice(sensor_name))
            self.proximity_sensors[i].enable(self.time_step)
       
        # Enable Ground Sensors
        self.left_ir = self.robot.getDevice('gs0')
        self.left_ir.enable(self.time_step)
        self.center_ir = self.robot.getDevice('gs1')
        self.center_ir.enable(self.time_step)
        self.right_ir = self.robot.getDevice('gs2')
        self.right_ir.enable(self.time_step)
        
        # Enable Emitter and Receiver (to communicate with the Supervisor)
        self.emitter = self.robot.getDevice("emitter") 
        self.receiver = self.robot.getDevice("receiver") 
        self.receiver.enable(self.time_step)
        self.receivedData = "" 
        self.receivedDataPrevious = "" 
        self.flagMessage = False
        
        # Fitness value (initialization fitness parameters once)
        self.fitness_values = []
        self.fitness = 0
        
        # 

    def check_for_new_genes(self):
        if(self.flagMessage == True):
                # Split the list based on the number of layers of your network
                part = []
                for n in range(1,len(self.number_neuros_per_layer)):
                    if(n == 1):
                        part.append((self.number_neuros_per_layer[n-1]+1)*(self.number_neuros_per_layer[n]))
                    else:   
                        part.append(self.number_neuros_per_layer[n-1]*self.number_neuros_per_layer[n])
                
                # Set the weights of the network
                data = []
                weightsPart = []
                sum = 0
                for n in range(1,len(self.number_neuros_per_layer)):
                    if(n == 1):
                        weightsPart.append(self.receivedData[n-1:part[n-1]])
                    elif(n == (len(self.number_neuros_per_layer)-1)):
                        weightsPart.append(self.receivedData[sum:])
                    else:
                        weightsPart.append(self.receivedData[sum:sum+part[n-1]])
                    sum += part[n-1]
                for n in range(1,len(self.number_neuros_per_layer)):  
                    if(n == 1):
                        weightsPart[n-1] = weightsPart[n-1].reshape([self.number_neuros_per_layer[n-1]+1,self.number_neuros_per_layer[n]])    
                    else:
                        weightsPart[n-1] = weightsPart[n-1].reshape([self.number_neuros_per_layer[n-1],self.number_neuros_per_layer[n]])    
                    data.append(weightsPart[n-1])                
                self.network.weights = data
                
                #Reset fitness list
                self.fitness_values = []
        
    def clip_value(self,value,min_max):
        if (value > min_max):
            return min_max;
        elif (value < -min_max):
            return -min_max;
        return value;

    def sense_compute_and_actuate(self):
        # MLP: 
        #   Input == sensory data
        #   Output == msotors commands
        output = self.network.propagate_forward(self.inputs)
        self.velocity_left = output[0]
        self.velocity_right = output[1]
        
        # Multiply the motor values by 3 to increase the velocities
        #print("input velocities : " + str(self.velocity_left*3) + ", " + str(self.velocity_right*3))
        self.left_motor.setVelocity(self.velocity_left*3)
        self.right_motor.setVelocity(self.velocity_right*3)

    def calculate_fitness(self):
        if self.i < 200:
            print(self.i)
        ### Define the fitness function to increase the speed of the robot and
        # Tabs to get the means of each component of the fitness
        self.forward_fitness_tab = [] 
        self.collision_fitness_tab = []
        self.spinning_fitness_tab = []
        
        # Coefficients to weights each fitness
        coeff_forward = 1.2
        coeff_collision = 1.4
        coeff_spinning = 0.4
        
        # Average for each fitness
        avg_f_fit = 0
        avg_c_fit = 0
        avg_s_fit = 0
        ### Forward fitness
        diffvel = abs(self.left_motor.getVelocity()-self.right_motor.getVelocity())
        if diffvel < 0.01:
            diffvel = 0.01
            
        forwardFitness = (self.left_motor.getVelocity()+self.right_motor.getVelocity())/(4*diffvel)
        
        if forwardFitness < 0:
            forwardFitness = 0
        else:
            forwardFitness = forwardFitness/(forwardFitness+1)

        self.forward_fitness_tab.append(coeff_forward*forwardFitness)
        #print("forward fitness" + str(forwardFitness))
                      
        ### Define the fitness function to avoid collision
        avoidCollisionFitness = 1
        n_collision = 0
        for i in range(3, len(self.inputs)):
          if self.inputs[i] > 0.6:
            avoidCollisionFitness /= (5*self.inputs[i])

        self.collision_fitness_tab.append(coeff_collision*avoidCollisionFitness)   
        #print("avoid collision fitnesss :" + str(avoidCollisionFitness))
        
        ### Define the fitness function to avoid spining behaviour
        spinningFitness = 6/diffvel
        spinningFitness = 1/(1 + 12*math.exp(-0.4*spinningFitness))
        self.spinning_fitness_tab.append(coeff_spinning*spinningFitness)
        #print("avoid spinning fitness : " + str(spinningFitness))
        
        ### Define the fitness function of this iteration which should be a combination of the previous functions         
        combinedFitness = (coeff_forward*forwardFitness + coeff_collision*avoidCollisionFitness + coeff_spinning*spinningFitness)/3
        #print("combined fitness : " + str(combinedsFitness))
        avg_f_fit = np.mean(self.forward_fitness_tab)
        avg_c_fit = np.mean(self.collision_fitness_tab)
        avg_s_fit = np.mean(self.spinning_fitness_tab)
        
        #if (self.i-200) % 1800 == 0:
        # Print the mean of each fitness only at the end of each try
            #print("forward : " + str(avg_f_fit))
            #print("collision : " + str(avg_c_fit))
            #print("spinning : " + str(avg_s_fit))
        self.fitness_values.append(combinedFitness)
        self.fitness = np.mean(self.fitness_values)

    def handle_emitter(self):
        # Send the self.fitness value to the supervisor
        data = str(self.number_weights)
        data = "weights: " + data
        string_message = str(data)
        string_message = string_message.encode("utf-8")
        #print("Robot send:", string_message)
        self.emitter.send(string_message)

        # Send the self.fitness value to the supervisor
        data = str(self.fitness)
        data = "fitness: " + data
        string_message = str(data)
        string_message = string_message.encode("utf-8")
        #print("Robot send fitness:", string_message)
        self.emitter.send(string_message)
            
    def handle_receiver(self):
        if self.receiver.getQueueLength() > 0:
            while(self.receiver.getQueueLength() > 0):
                # Adjust the Data to our model
                self.receivedData = self.receiver.getData().decode("utf-8")
                self.receivedData = self.receivedData[1:-1]
                self.receivedData = self.receivedData.split()
                x = np.array(self.receivedData)
                self.receivedData = x.astype(float)
                #print("Controller handle receiver data:", self.receivedData)
                self.receiver.nextPacket()
                
            # Is it a new Genotype?
            if(np.array_equal(self.receivedDataPrevious,self.receivedData) == False):
                self.flagMessage = True
                
            else:
                self.flagMessage = False
                
            self.receivedDataPrevious = self.receivedData 
        else:
            #print("Controller receiver q is empty")
            self.flagMessage = False

    def run_robot(self):        
        # Main Loop
        self.i = 0
        while self.robot.step(self.time_step) != -1:
            # This is used to store the current input data from the sensors
            self.inputs = []
            
            # Emitter and Receiver
            # Check if there are messages to be sent or read to/from our Supervisor
            self.handle_emitter()
            self.handle_receiver()
            
            # Read Ground Sensors
            left = self.left_ir.getValue()
            center = self.center_ir.getValue()
            right = self.right_ir.getValue()
            #print("Ground Sensors \n    left {} center {} right {}".format(left,center,right))
                        
            ### Please adjust the ground sensors values to facilitate learning 
            min_gs = 300
            max_gs = 900
            
            if(left > max_gs): left = max_gs
            if(center > max_gs): center = max_gs
            if(right > max_gs): right = max_gs
            if(left < min_gs): left = min_gs
            if(center < min_gs): center = min_gs
            if(right < min_gs): right = min_gs
            
            # Normalize the values between 0 and 1 and save data
            self.inputs.append((left-min_gs)/(max_gs-min_gs))
            self.inputs.append((center-min_gs)/(max_gs-min_gs))
            self.inputs.append((right-min_gs)/(max_gs-min_gs))
            #print("Ground Sensors \n    left {} center {} right {}".format(self.inputs[0],self.inputs[1],self.inputs[2]))
            
            # Read Distance Sensors
            for i in range(8):
                ### Select the distance sensors that you will use
                if(i==0 or i==1 or i==2 or i==3 or i==4 or i==5 or i==6 or i==7):        
                    temp = self.proximity_sensors[i].getValue()
                    
                    ### Please adjust the distance sensors values to facilitate learning 
                    min_ds = 200
                    max_ds = 800
                    
                    if(temp > max_ds): temp = max_ds
                    if(temp < min_ds): temp = min_ds
                    
                    # Normalize the values between 0 and 1 and save data
                    self.inputs.append((temp-min_ds)/(max_ds-min_ds))
                    #print("Distance Sensors - Index: {}  Value: {}".format(i,self.inputs[i+3]))
    
            # GA Iteration       
            # Verify if there is a new genotype to be used that was sent from Supervisor  
            self.check_for_new_genes()
            # Define the robot's actuation (motor values) based on the output of the MLP
            self.sense_compute_and_actuate()
            # Calculate the fitnes value of the current iteration
            self.calculate_fitness()
            self.i += 1
            
            # End of the iteration 
            
if __name__ == "__main__":
    # Call Robot function to initialize the robot
    my_robot = Robot()
    # Initialize the parameters of the controller by sending my_robot
    controller = Controller(my_robot)
    # Run the controller
    controller.run_robot()
    
