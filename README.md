# Evolutionary Algorithm-Trained MLP Classifier

This project implements a **Multi-Layer Perceptron (MLP)** with one hidden layer, trained using an **evolutionary algorithm** instead of traditional backpropagation. The model classifies 2D data points into binary classes using a step activation function at the output.

## Features

* Manually defined MLP structure with 2 hidden neurons and 1 output neuron
* Uses **tanh activation** in the hidden layer and **step function** in the output
* Trains the MLP using an **evolutionary algorithm**:

  * Random weight population initialization
  * Fitness evaluation based on classification accuracy
  * Crossover for reproduction
  * No mutation used (can be added)

## Dataset

A synthetic dataset of 2D inputs:

```python
trainInput = [[1, 1], [9.4, 6.4], [2.5, 2.1], [8, 7.7], [0.5, 2.2],
              [7.9, 8.4], [7, 7], [2.8, 0.8], [1.2, 3], [7.8, 6.1]]
trainOutput = [1, -1, 1, -1, -1, 1, -1, 1, -1, -1]
```

## Project Structure

* `Neuron`: Represents a neuron in the MLP with methods for forward propagation
* `Multi_layer_perceptron`: Contains two hidden neurons connected to one output neuron
* `population_generation()`: Initializes random weights for the population
* `fitness_evaluation()`: Evaluates classification accuracy of individuals
* `crossover()`: Performs 1D slicing crossover between two individuals
* `next_generation()`: Selects top individuals and generates new offspring

## Training (via Evolution)

Training is done using evolutionary logic (commented out in code for reuse):

```python
first_population = population_generation(1000)
first_fitness = fitness_evaluation(first_population)
first_selection = natural_selection(first_fitness)
new_generation = next_generation(first_selection, first_population)

for i in range(500):
    fitness = fitness_evaluation(new_generation)
    selection = natural_selection(fitness)
    best_weight_fitness = selection[list(selection.keys())[0]]
    if best_weight_fitness == 100:
        trained_weights.append(new_generation[list(selection.keys())[0]])
        break
    new_generation = next_generation(selection, new_generation)
```

## Example Trained Weights

The repo includes 4 pre-trained weight sets for reproducibility. Fitness evaluation for each set is printed using:

```python
for i in range(4):
    fitness = fitness_evaluation(trained_weights_sets[i])
    print(fitness)
```

## Getting Started

1. Clone the repo:

   ```bash
   git clone https://github.com/your-username/multilayer-perceptron.git
   cd multilayer-perceptron
   ```
2. Run the script:

   ```bash
   python multi_layer_perceptron.py
   ```
