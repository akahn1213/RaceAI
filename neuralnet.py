"""
Computes the acceleration and steering given a chromosome and input node activations

This neural network is a multi layer perceptron. 
It takes inputs as a list of values which do NOT need to be normalized.
The activation function of each neuron is a sigmoid.

There also exists here an implementation of a genetic algorithm, which takes a 1D array of values between -2 and +2, called a chromosome
The chromosome's elements (genes), and the values of those elements (alleles), are then mapped to the weights and biases of the perceptron.
In the case of the race-car AI, the last 3 genes are used to determine the color of the car. One value for Red, Green, and Blue.

Only the topology of the perceptron needs to be set by the user. The mapping between chromosome elements and weights is done automatically and determined by the topology.
To use this for a particular learning application, all that needs to be done is to initialize a population with a random chromosome, and then input that chromosome alongside a fitness value into the genetic operators to produce new offspring

The genetic operations which facilitate learning are as follows:

1. Selection
    The 5 best performing (highest fitness) members of the population, each represented as one chromosome, are selected. 
    If there is no overall best performer, then set the highest performing chromosome to be the overall best performer. This chromosome will be saved between generations, unless another chromosome with better fitness dethrones it.
    If there is already an overall best performer determined from an earlier generation, choose the overall best performer and the top 4 performers of the current population instead.
    These 5 chromosomes are then moved to the crossover(recombination) step
    
2. Crossover
    This step generates new offspring with traits similar to its parents
    Given 5 chromosomes, there are 10 unique pairs that can be made among them. The resulting children of these pairs (parents) become the next generation
    Parents are crossed-over as follows:
        For each gene, find the interval between the two parents' alleles. Say for example, parent 1 has a value of -1.5 and parent 2 has a value of 0.5
        Find the relative fitness (fitness ratio) of the parents. If parent 2 has a fitness of 200 and parent 1 has a fitness of 100, then this relative fitness is 2/3. It is designed in terms of parent 2
        Find the value that is between the two parents' alleles, scaled by the fitness ratio. In this example, that resulting value is (0.5 - (-1.5))*2/3 + (-1.5) = -0.167
            This is done so that this child's chromosome is more closer to the better parent than the worse one
        Find a random value, centered at this resulting intermediate value (-0.167), within the range of +/- 10% of the interval between the parents' alleles
            In this case, since ths interval between the parents' alleles is 2.0, then the resulting child's allele for this gene will be a random number in the range of -0.167 +/- 0.2
    After all genes have been processed in this manner, the chromosome is then passed on to the mutation step
    
3. Mutate
    This step blindly changes the value of genes, so that new behaviors could be introduced into the population. In other words, this allows the solution space to jump to different local maxima, instead of converging along only one local maximum, which is what step 2 is all about
    Given the fitness of the parents, generate a probability of mutation that is inversely proportional to the best of the two parents' fitnesses
    Then, iterate through all of the child's genes, and roll a random number between 0 and 1. If it is less than the probability of mutation, set that gene's allele to a new random value between -2 and 2
    
    
The result is a new set of 10 chromosomes which have closer behavior to the best performing member of the previous generation, and hopefully is better at performing the desired task
"""
import numpy as np

bias_weight = 0 #Strength of the biases. 0 to disable biases


#Configure the topology of the perceptron
n_inputs = 8
n_outputs = 2
n_hidden_layers = 1
n_hidden_neurons = 4
if(n_hidden_layers > 0):
    n_genes = np.square(n_hidden_neurons)*(n_hidden_layers - 1) + n_hidden_neurons*(n_hidden_layers + n_inputs + n_outputs) + n_outputs #Biases
else:
    n_genes = n_inputs*n_outputs + n_outputs #Biases

inputs = np.empty(n_inputs)
layers = np.empty(shape = [n_hidden_layers, n_hidden_neurons])
outputs = np.empty(n_outputs)


#Return an array of length n_genes + 3 (weights + 3 colors), where each element is a random number between -2 and +2
def generate_random_chromosome():
    return 4*np.random.rand(n_genes+3) - 2

overall_best_chromosome = generate_random_chromosome()
overall_best_fitness = 0.


def sigmoid(x):
    return 1. / (1. + np.exp(-x))
    
def get_pm(x): #Mutation Probability
    return 1. / (1. + np.exp((-2*np.log(1/9))*x))

#The Neural Net. Given the inputs, and the weights and biases (encoded in the chromosome), produce the outputs
def compute(a1, a2, a3, a4, a5, a6, a7, a8, chromosome):
    inputs = np.asarray([a1, a2, a3, a4, a5, a6, a7, a8])
    if(n_hidden_layers > 0):
        #Hidden layer 1
        for n in range(n_hidden_neurons):
            start = n*(n_inputs + 1)
            layers[0][n] = sigmoid(np.dot(inputs, chromosome[start:start+n_inputs]) + bias_weight*chromosome[start+n_inputs])
        #Rest of the hidden layers
        for layer in range(1, n_hidden_layers):
            for n in range(n_hidden_neurons):
                start = np.square(n_hidden_neurons)*(layer-1) + n_hidden_neurons*(n_inputs + layer + n) + n
                layers[layer][n] = sigmoid(np.dot(layers[layer-1], chromosome[start:start+n_hidden_neurons]) + bias_weight*chromosome[start+n_hidden_neurons])
        #Outputs
        start = np.square(n_hidden_neurons)*(n_hidden_layers-1) + n_hidden_neurons*(n_inputs + n_hidden_layers)
        outputs[0] = sigmoid((5)*(np.dot(layers[n_hidden_layers - 1], chromosome[start:start+n_hidden_neurons]) + bias_weight*chromosome[start+n_hidden_neurons]))
        outputs[1] = sigmoid((5)*(np.dot(layers[n_hidden_layers - 1], chromosome[start+n_hidden_neurons+1:start+2*n_hidden_neurons+1]) + bias_weight*chromosome[start+2*n_hidden_neurons+1]))
    else:
        outputs[0] = sigmoid((1/100)*np.dot(inputs, chromosome[0:n_inputs]) + bias_weight*chromosome[n_inputs])   
        outputs[1] = sigmoid((1/100)*(np.dot(inputs, chromosome[n_inputs+1:2*n_inputs+1]) + bias_weight*chromosome[2*n_inputs+1]))
    
    o1 = -1 + np.round(2*outputs[0])
    o2 = -1 + np.round(2*outputs[1])
    accel = np.absolute(o1)*(0.18*o1 - 0.12)
    steering = 5*o2    
    return accel, steering

    
    
def rand_between(num1, num2):
  high = num1
  low = num2
  if (num2>num1):
    high = num2
    low = num1
  return np.random.rand()*(high - low) + low

def get_skewed_value(v1, v2, f1, f2):
    fitness_ratio = 0
    if(f1 > 0 and f2 > 0):
        fitness_ratio = f2/(f1+f2)
    elif(f1 > 0 and f2 < 0):
        fitness_ratio = f2/(f2 - f1)
    elif(f1 < 0 and f2 > 0):
        fitness_ratio = f2/(f2 - f1)
    elif(f1 < 0 and f2 < 0):
        fitness_ratio = ((f1+f2) - f2)/(f1+f2)
    elif(f1 == 0 and f2 > 0):
        fitness_ratio = 1
    elif(f1 == 0 and f2 < 0):
        fitness_ratio = 0
    elif(f1 > 0 and f2 == 0):
        fitness_ratio = 0
    elif(f1 < 0 and f2 == 0):
        fitness_ratio = 1
    else:
        fitness_ratio = (1/2)
        
    skewed_value = v1 + fitness_ratio*(v2-v1) + 0.2*np.absolute(v2-v1)*(np.random.rand() - 0.5)
    return skewed_value

def crossover(parent1, parent2, fitness1, fitness2):  
  child = np.empty(n_genes+3)
  for gene in range(n_genes+3):
    child[gene] = get_skewed_value(parent1[gene], parent2[gene], fitness1, fitness2)
  return child


def mutate(chromosome, fitness, rel_fitness):
  mutated_chromosome = chromosome

  if(fitness < 5200):
    p_m = get_pm(fitness/5200)
  else:
    p_m = 0.001*(1/rel_fitness)
    
  for gene in range(n_genes+3):
    mutator = np.random.rand()
    if(mutator < p_m): mutated_chromosome[gene] = 4*np.random.rand() - 2
  return mutated_chromosome

def select(chromosomes, fitnesses, n_population = 10):
  global overall_best_chromosome
  global overall_best_fitness
  n_parents = int(np.ceil(n_population/2))
  sorted_chromosomes = np.empty(shape = [n_parents, n_genes+3])
  sorted_fitnesses = np.empty(n_parents)
  selected_chromosomes = np.empty(shape = [n_parents, n_genes+3])
  selected_fitnesses = np.empty(n_parents)
  indeces = (-np.asarray(fitnesses)).argsort()[:n_parents]
  sorted_chromosomes = np.asarray(chromosomes)[indeces]
  sorted_fitnesses = np.asarray(fitnesses)[indeces]
  
  prev_overall = overall_best_fitness
  
  #Check for best fitness
  if(sorted_fitnesses[0] > overall_best_fitness):
    selected_chromosomes = sorted_chromosomes
    selected_fitnesses = sorted_fitnesses
    overall_best_chromosome = sorted_chromosomes[0]
    overall_best_fitness = sorted_fitnesses[0]
  else:
    selected_chromosomes[0] = overall_best_chromosome
    selected_fitnesses[0] = overall_best_fitness
    for i in range(1, n_parents):
        selected_chromosomes[i] = sorted_chromosomes[i-1]
        selected_fitnesses[i] = sorted_fitnesses[i-1]
        
  print("Overall Best Fitness: " +str(np.round(overall_best_fitness, 1))+", Previous Highest: "+str(np.round(sorted_fitnesses[0], 1)))
  return selected_chromosomes, selected_fitnesses

  
  
def get_new_generation(chromosomes, fitnesses):

    #Calculate total fitness of population
    if(np.min(fitnesses) < 0): total_fitness = np.sum(np.array(fitnesses) - np.min(fitnesses) + 1)
    else: total_fitness = np.sum(np.array(fitnesses))

    #Select
    parents, parent_fitnesses = select(chromosomes, fitnesses)
    
    #Crossover
    children = []
    children_fitnesses = []
    for i in range(4):
        for j in range(i+1, 5):
            #Make children
            children.append(crossover(parents[i], parents[j], parent_fitnesses[i], parent_fitnesses[j]))
            children_fitnesses.append(np.max(np.asarray(parent_fitnesses[i], parent_fitnesses[j])))
            
    #Mutate
    mutated_children = []
    for child in range(len(children)):
        mutated_children.append(mutate(children[child], children_fitnesses[child], children_fitnesses[child]/total_fitness))
    np.random.shuffle(mutated_children)
    return mutated_children 


def import_chromosomes(timestamp):
    chromosomes = []
    for i in range(10):
        chromosomes.append(np.load('chromosomes/'+timestamp+'/chromosome_'+str(i)+'.npy'))
    return chromosomes

    
    

