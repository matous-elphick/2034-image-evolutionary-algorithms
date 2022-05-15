"""
NAME
    Genetic Algorithms to Recreate Target Images Using Polygons
DESCRIPTION
    Genetic algorithm that reconstructs a user defined target image using
    a combination of solution representation, mutation, selection and crossover.

    This is the selection variant algorithm, which is a variant of how
    individuals are selected

    Solution Representation:
        - Initially 10 three vertex polygons in an individual
    Mutation:
        - Add polygon to individual
        - Mutate colour of polygon in individual
        - Mutate position of polygon in individual
        - Shuffle indexes of polygons in individual
    Selection:
        - Tournament selection
    Crossover:
        - Position based crossover
FILE
    reference.py
METHODS & FUNCTIONS
    - initialize_polygon() : Initialises individual polygon
    - initialize() : Initialises list of polygons
    - draw() : Generates image using the population
    - evaluate() : Evaluates the populations and returns the fitness
    - select() : Selects two individuals from the population
    - median_fitness() : Calculates the median fitness from a population
    - combine() : Combines two individuals to create a new child
    - mutate() : Mutates the chromosome of an individual from a population
    - centroid() : Calculates the centre point of a polygon
    - draw_median_fitness_graph() : Plots a graph of the median fitness over generations
    - read_config() : Gets the parameters from a config JSON file
    - update_target_image() : Updates the target image for the reconstruction
    - run() : Main method for running the genetic algorithms
GLOBAL VARIABLES
    - MAX_POLYGON_COUNT : Maximum number of polygons per individual
    - POLYGON_COUNT : Starting number of polygons for an individual
    - BACKGROUND_COLOUR : Background colour of the image reconstruction
    - MAX : Number of bits for the image
    - TARGET : Target image to be reconstructed
"""
import copy
import os
import json
import argparse
import random
import time
import matplotlib.pyplot as plt
import numpy as np
from evol import Evolution, Population
from PIL import Image, ImageDraw, ImageChops
from deap import tools

MAX_POLYGON_COUNT = 100  # Number of polygons 100 working
POLYGON_COUNT = 10  # Number of polygons 100 working
BACKGROUND_COLOUR = "black"
MAX = 255 * 200 * 200  # Number of bits for the image
TARGET = Image.open("target-images/darwin.png")  # Target Image to be reconstructed
TARGET.load()  # Read image and close the file


def initialize_polygon():
    """
    Creates a polygon with random values for the polygons RGBA and point coordinates.

    Uses the random function to select a random integer between 0,255 for RGB
    and 0,200 for point coordinates.

    :return: A list containing four tuples, one for colour and three for the points coordinates.
    """
    return [(random.randint(0, 255),  # Red Value
             random.randint(0, 255),  # Green Value
             random.randint(0, 255),  # Blue Value
             random.randint(15, 80)),  # Alpha Value 60-80
            (random.randint(0, 200),  # Point 1 X Value
             random.randint(0, 200)),  # Point 1 Y Value
            (random.randint(0, 200),  # Point 2 X Value
             random.randint(0, 200)),  # Point 2 Y Value
            (random.randint(0, 200),  # x3
             random.randint(0, 200))]  # y3


def initialize():
    """
    Creates a list of polygons for the population for the evolutionary algorithm

    :return: A list of polygons with a length set to POLYGON COUNT
    """
    return [initialize_polygon() for i in range(POLYGON_COUNT)]


def draw(individual, save=False, generation=-1, path=None):
    """
    Creates the image using the polygons within the population

    :param generation: Generation number for the image file name
    :param path: The path to save the new image to
    :param individual: List of polygons representing the individual
    :param save: Value to indicate if the draw function should save the image to local directory
    :return: An image containing the populations polygons
    """

    # Creating the Image and Canvas to store the polygon representation
    image = Image.new("RGB", (200, 200),color=BACKGROUND_COLOUR)
    canvas = ImageDraw.Draw(image, "RGBA")

    for polygon in individual:
        # Looping through the population and adding it to the canvas
        canvas.polygon(polygon[1:], fill=polygon[0])

    if save:
        image.save(f"{path}/Best.png")  # Saves the image if save is True
    if save and generation > -1:
        image.save(f"{path}/Generation_{generation}.png")
    return image


def evaluate(individual):
    """
    Evaluates the populations and returns the fitness of the population based on the
    pixel difference

    :param individual: List of polygons representing the individual
    :return: The fitness float value between 0 and 1
    """

    # Creating an image representation of the population
    image = draw(individual)
    # Getting the pixel-by-pixel difference between the two
    diff = ImageChops.difference(image, TARGET)
    hist = diff.convert("L").histogram()  # Creating a histogram based on the difference
    count = sum(i * n for i, n in enumerate(hist))
    return (MAX - count) / MAX


def select(population):
    """
    Selecting existing individuals to be used for the new population.

    :param population: List of individuals representing the population
    :return: A list containing two individuals
    """

    mother_group = random.choices(population, k=5)
    father_group = random.choices(population, k=5)

    mother_group = sorted(mother_group, key=lambda agent: agent.fitness, reverse=True)
    father_group = sorted(father_group, key=lambda agent: agent.fitness, reverse=True)

    return mother_group[0], father_group[0]


def median_fitness(population):
    """
    Getting the median fitness value from a population

    :param population: List of individuals representing the population
    :return: The median fitness value as a float
    """
    population_length = len(population)
    if population_length % 2 == 0:  # If length of population is even
        median1 = population[population_length // 2].fitness
        median2 = population[population_length // 2 - 1].fitness
        return (median1 + median2) / 2
    return population[population_length // 2].fitness


def combine(*parents):
    """
    Combines the chromosomes of two individuals and creates a new individual
    All the polygons with a centre point less than 100 are from the first parent
    All the polygons with a centre point greater than 100 are from the second parent

    :param parents: Two individuals to be used to create the new individual
    :return: A new individual of type list
    """

    child = []

    for polygon in parents[0]:
        x_coord, y_coord = centroid(polygon[1:])
        if len(child) < 100 and x_coord < 100:
            child.append(copy.deepcopy(polygon))
    for polygon in parents[1]:
        x_coord, y_coord = centroid(polygon[1:])
        if len(child) < 100 and x_coord > 100:
            child.append(copy.deepcopy(polygon))

    # Returns one of the new individuals at random
    return child


def mutate(individual, rate):
    """
    Mutates a polygon of an individual in three ways:
        - Adds a new polygon to an individual if the number of polygons
            is less than 100
        - Mutate the points of a random polygon in the individual
        - Mutate the colours of a random polygon in the individual

    :param individual: List of polygons representing the individual
    :param rate: Rate of mutation
    :return: The individual after mutations
    """
    roulette = random.random()
    polygon_index = random.randint(0, len(individual) - 1)
    if roulette < 0.25:
        if len(individual) < MAX_POLYGON_COUNT:
            individual.append(initialize_polygon())
    elif 0.25 < roulette < 0.5:
        # mutate points
        polygon = individual[polygon_index]
        coords = [x for point in polygon[1:] for x in point]
        tools.mutGaussian(coords, 0, 10, 0.25)  # 3
        coords = [max(0, min(int(x), 200)) for x in coords]
        polygon[1:] = list(zip(coords[::2], coords[1::2]))
    elif 0.5 < roulette < 0.75:
        # mutate colours
        colours = list(individual[polygon_index][0])
        tools.mutGaussian(colours, 0, 10, 0.25)  # 3
        colours = [max(0, min(int(x), 255)) for x in colours]
        individual[polygon_index][0] = tuple(colours)
    else:
        tools.mutShuffleIndexes(individual, 0.01)
    return individual


def centroid(vertexes):
    """
      Calculates the centre point of a polygon

      :param vertexes: List of coordinates of the points of a polygon
      :return: X and Y coordinates of the centre point of the polygon
      """
    # Getting all the x coord from polygon
    x_list = [vertex[0] for vertex in vertexes]
    # Getting all the y coord from polygon
    y_list = [vertex[1] for vertex in vertexes]
    length = len(vertexes)  # Number of points in polygon
    x_coord = sum(x_list) / length  # X centre point of polygon
    y_coord = sum(y_list) / length  # Y centre point of polygon
    return x_coord, y_coord


def draw_median_fitness_graph(fitness_array):
    """
    Creates a graph of the median fitness for each generation
    using the matplotlib

    :param fitness_array: List of median fitness's for each generation
    """
    y_points = np.array(fitness_array)

    plt.plot(y_points)
    plt.title("Evolutionary Algorithm To Reconstruct Images Using Polygons")
    plt.xlabel("Generations")
    plt.ylabel("Fitness")

    plt.show()


def read_config(arg):
    """
    Reads the configuration parameters for the evolutionary
    algorithm from a user inputted JSON file

    :param arg: File path for the JSON file
    :return: Parameters for the evolutionary algorithm
    """
    with open(arg.filename, mode='r') as json_data:
        return json.load(json_data)


def update_target_image(target_image_filepath):
    """
    Updates the global variable TARGET which is the
    target image

    :param target_image_filepath: The file path of the target image
    """
    global TARGET
    TARGET = Image.open(target_image_filepath)


def run(population=75, survival_rate=0.1, mutation_rate=0.01, generations=2000,
        target_image="test-images/darwin.png"):  # 2000
    """
    The main function for running the algorithm.

    Runs the evol algorithm using the user inputted data from the json
    file or the default values.

    :param target_image: File path for target image
    :param population: The population size
    :param survival_rate: The survival rate as a float value
    :param mutation_rate: The rate of mutation as a float value
    :param generations: The number of generations to run the evol algorithm
    """
    fitness_array = []
    best_fitness = 0
    best_population = None
    best_generation_number = 0

    if target_image != "test-images/darwin.png":
        update_target_image(target_image)

    path = os.path.join(os.getcwd(), "v3output")

    population = Population.generate(initialize, evaluate, size=population, maximize=True)

    evo1 = (Evolution().survive(fraction=survival_rate)
            .breed(parent_picker=select, combiner=combine)
            .mutate(mutate_function=mutate, rate=mutation_rate, elitist=True)
            .evaluate())

    for i in range(generations):
        population = population.evolve(evo1)
        if population.current_best.fitness > best_fitness:
            best_fitness = population.current_best.fitness
            best_population = population
            best_generation_number = i

        if i % 1 == 0:
            print("i =", i, " best =", population.current_best.fitness,
                  " median =", median_fitness(population=population),
                  " worst =", population.current_worst.fitness,
                  " best pol count =", len(population.current_worst.chromosome))
        if i % 10 == 0 or i == 0:
            if not os.path.exists(path):
                os.mkdir(path)
            draw(population.current_best.chromosome, True, i, path)

        fitness_array.append(median_fitness(population=population))

        if population.current_best.fitness - (population.current_best.fitness % 0.95):
            draw(population.current_best.chromosome, True, i, path)
            exit()

    draw(best_population.current_best.chromosome, True, path=path)
    print()
    print("Best Fitness: ", best_fitness, "Generation Number: ", best_generation_number)

    draw_median_fitness_graph(fitness_array)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--filename", required=False, type=str)
    args = parser.parse_args()
    start_time = time.time()

    if args.filename is not None:
        params = read_config(args)
        print(params)
        run(population=params["population"],
            survival_rate=params["survival_rate"],
            mutation_rate=params["mutation_rate"],
            generations=params["generations"],
            target_image=params["target_image"])
    else:
        run()
    print("--- %s seconds ---" % (time.time() - start_time))
