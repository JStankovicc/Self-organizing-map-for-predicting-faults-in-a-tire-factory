# Kohonen Self-Organizing Map for Automotive Tire Production

## Task Description
In the process of manufacturing automotive tires, the components constituting the tire may exhibit imperfections that affect its usability. Anomalies, represented by properties x1, x2, and x3, have been collected during the tire production process. However, understanding the interrelation of these properties has proven challenging for the team of scientists and engineers. The task is to create a Kohonen self-organizing map with 16 neurons to detect similarities and correlations between these variables, ultimately aiming for the classification of defective instances.

### Training
Train the Kohonen self-organizing map using the input dataset in `hw2input.csv`.

Using the training results, instances 1-20, 21-60, and 61-120 are confirmed to share common characteristics, representing three different classes - A, B, and C (respectively).

## Exercises
1. **Implement the Kohonen self-organizing map**: Develop a program that automates the entire process, from data input to result display.
2. **Identify Neuron Sets**: Determine neuron sets 1-16 corresponding to classes A, B, and C.
3. **Classify Test Data**: For examples in the test file `hw2test.csv`, determine the class to which each instance belongs.

### 1.1 Documentation

### How to Run the Program
1. **Language**: Python
2. **External Libraries**: NumPy
3. **Configuration**: No specific setup needed
4. **Compilation and Execution**: Run `main.py` using the command: `python main.py`

Ensure the Python environment is set up with the required dependencies installed.

## Output Format
The program will generate an output file in the same format as the example provided in the exercises.

