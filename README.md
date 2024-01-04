# # Inteligência Artificial Aplicada a Jogos

Bernardo Neves - 23494

---
## Requirements
- Python 3.9

---
## Setting up

``` shell
pip install -r requirements.txt
```

---
## Running

``` shell
pygame -u survive.py
```

---
## Controls

- **BFS**: Toggles Breadth-First Search 
- **Dijkstra**: Toggles Dijsktra
- **A star**: Toggles A*
- **Test**: Toggles neat-python testing and visualization
- **Train**: Closes the pygame window and starts neat-python training, once done re-opens pygame
- **Follow**: Toggles player following current active path
- **Focus**: Moves the camera so that the player is always in view
- **Goal**: Generates a new goal at a random position on the map
- **Map**: Generates a new map
- KeyBinds:
	- "A": Move camera left
	- "W": Move camera up 
	- "S": Move camera  down
	- "D": Move camera right

---
## Description
This Python game project focuses on showcasing the NeuroEvolution of Augmenting Topologies (NEAT) through a dynamic visualization tool. The central aspect of the project is the integration of the NEAT model, allowing users to observe the learning process and decision-making abilities of an evolving neural network.

The game scenario centers around a procedurally generated map created using a Random Walk Algorithm. The NEAT model is trained to navigate this map, making directional decisions based on the presence of food and water to ensure its survival. Users can analyze the NEAT model's performance in terms of hunger, thirst, movements, and overall survival time, providing insights into its adaptive capabilities.

Moreover, the project incorporates three fundamental graph search algorithms—Breadth-First Search, Dijkstra, and A*. These algorithms play a role in the NEAT model's pathfinding between a player and a goal, enhancing the overall complexity of the gaming experience.

Additionally, this project provides users with the capability to compare the path lengths and execution times of three fundamental graph search algorithms—Breadth-First Search, Dijkstra, and A*. As the NEAT model calculates paths between a player and a goal, users can analyze and contrast the efficiency of these graph algorithms in different scenarios. This comparative analysis adds an educational dimension to the project, enabling users to deepen their understanding of both the NEAT model's adaptive learning and the computational intricacies involved in traditional graph search methods. By incorporating these elements, the project offers a holistic exploration of artificial intelligence and graph algorithms within a dynamic and interactive gaming environment, making it a valuable contribution to the intersection of AI and game development.

## AI techniques used
### Map
### Random Walk Algorithm
- **Objective:**
    - The primary objective of a random walk is exploration and sampling rather than reaching a specific target or finding an optimal path.
- **Initialization:**
    - Start at an initial node in the graph.
- **Steps:**
    - At each step, randomly choose one of the neighboring nodes to move to. The choice can be uniform or biased based on certain probabilities.
    - Repeat this process for a specified number of steps or until a particular condition is met.
- **Termination:**
    - The random walk can terminate based on a predefined number of steps, reaching a certain node, or other specified criteria.
- **Applications:**
    - Random walks are used in various fields such as physics, computer science, and finance for simulation and modeling.
    - In graph theory, random walks can be used to study properties of graphs, like connectivity and convergence.
    - In machine learning, random walks can be employed for generating synthetic data or exploring state spaces.
- **Biased Random Walks:**
    - Random walks can be biased by assigning probabilities to different edges or nodes. For example, a biased random walk might prefer certain directions or nodes over others.
- **Limitations:**
    - Random walks may not guarantee coverage of all nodes, especially in large or sparse graphs.
    - The lack of a specific objective means that random walks may not be suitable for tasks that require precise solutions or optimization.

### Pathfinding
#### Breadth First Search 
- **Objective:** BFS is an algorithm for traversing or searching tree or graph data structures. It doesn't prioritize finding the shortest path but systematically explores all nodes at the current depth before moving on to the next depth level.
- **Initialization:** Start at the source node and enqueue it in a queue. Mark it as visited.
- **Expansion:** Explore all neighbors of the current node, enqueue unvisited neighbors, and mark them as visited.
- **Termination:** Continue until the destination node is found or the entire graph is traversed.
- **Result:** BFS guarantees the shortest path in an unweighted graph but is less efficient for finding the shortest path in weighted graphs compared to Dijkstra's or

#### Dijsktra
- **Objective:** Dijkstra's algorithm is designed for finding the shortest path from a source node to all other nodes in a weighted graph.
- **Initialization:** Assign a distance value to each node, initially set to infinity for all nodes except the source, which is set to 0. Maintain a set of unvisited nodes.
- **Selection:** Repeatedly choose the unvisited node with the smallest distance, update the distances of its neighbors, and mark it as visited.
- **Termination:** Stop when all nodes have been visited or when the destination node is reached.
- **Result:** Dijkstra's algorithm guarantees the shortest path to each node from the source.

#### A*
- **Objective:** A* (pronounced "A star") is a heuristic search algorithm used for finding the shortest path from a source node to a destination node in a graph.
- **Initialization:** Similar to Dijkstra's, initialize distance values and maintain a set of unvisited nodes.
- **Selection:** Prioritize nodes based on a combination of the actual cost from the source and a heuristic estimate of the cost to the destination.
- **Termination:** Stop when the destination node is reached.
- **Result:** A* algorithm is more informed than Dijkstra's by using heuristics, allowing it to efficiently explore paths likely to lead to the optimal solution.
### Neat-Python
**NeuroEvolution of Augmenting Topologies** (**NEAT**) is a [genetic algorithm](https://en.wikipedia.org/wiki/Genetic_algorithm "Genetic algorithm") (GA) for the generation of evolving [artificial neural networks](https://en.wikipedia.org/wiki/Artificial_neural_network "Artificial neural network") (a [neuroevolution](https://en.wikipedia.org/wiki/Neuroevolution "Neuroevolution") technique) developed by [Kenneth Stanley](https://en.wikipedia.org/wiki/Kenneth_Stanley "Kenneth Stanley") and [Risto Miikkulainen](https://en.wikipedia.org/wiki/Risto_Miikkulainen)

#### Genetic
- **Initialization:** A population of neural networks is created, each represented by a set of genes encoding its structure and parameters.
- **Evaluation:** The fitness of each neural network in the population is evaluated based on its performance in the given task.
- **Selection:** Neural networks are selected for reproduction based on their fitness. Higher fitness increases the chances of being selected.
- **Crossover (Crossover):** Pairs of neural networks are combined to create offspring by exchanging genetic information, such as nodes and connections, to explore new network structures.
- **Mutation:** Random changes are applied to the genes of some neural networks to introduce diversity and potentially improve performance.
- **Speciation:** To maintain diversity in the population, neural networks are grouped into species based on similarity. This helps prevent convergence to a suboptimal solution.

#### Neural Network
- **Representation:** Neural networks in NEAT are represented as directed graphs where nodes represent neurons, and connections represent synapses between neurons. Each connection has associated weights.
- **Node and Connection Genes:** Nodes and connections are represented as genes in the genome of a neural network. The genome specifies the architecture and parameters of the neural network.
- **Activation:** During evaluation, the neural network processes inputs through its architecture, applying activation functions to produce outputs.
- **Fitness Evaluation:** The performance of the neural network is assessed based on a fitness function that measures how well it accomplishes the task at hand.
- **Adaptation:** Through the genetic algorithm, the neural network's architecture and parameters are adapted over generations, leading to the evolution of more effective networks for the given task.