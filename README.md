# 2034-image-evolutionary-algorithms
Genetic algorithm that reconstructs a user defined target image using polygons and
a combination of solution representation, mutation, selection and crossover.
Evol was used as the frame work for the genetic algorithm.

There are four different genetic algorithms, a reference algorithm which is the 
general basis and three variants which provide alternatives for the solution representation,
offspring generation and selection of the 

## Installation
To install the required packages for the program to work use the following
command in the shell of your venv.

```python
pip install -r requirements.txt
```
## Running
For running any of the files:
  - reference.py
  - variant1.py
  - variant2.py
  - variant3.py

One has two options for running them:
  - Default
  - Custom
### Default Run
The default run has the variables ```population``` , ```survival_rate```,
```mutation_rate```, ```generations``` and ```target_image``` preset so all 
the user has to do is the following.
```bash
python reference.py
```
```bash
python variant1.py
```
```bash
python variant2.py
```
```bash
python variant3.py
```
### Custom Run
The custom run allows the user to predefine the variables ```population``` , 
```survival_rate```,```mutation_rate```, ```generations``` and ```target_image```
using a JSON file. The following is an example JSON file.
```json
{
  "population":300,
  "survival_rate":0.1,
  "mutation_rate":0.005,
  "generations":150,
  "target_image":"test-images/cat.png"
}
```
To run the genetic algorithm with the JSON one must do the following.
```bash
python <reference|variant1|variant2|variant3>.py -f init.json
```
OR
```bash
python <reference|variant1|variant2|variant3>.py --filename init.json
```
This will work with any of the algorithms for example:
```bash
python reference.py -f init.json
python variant1.py -f init.json
python variant2.py -f init.json
python variant3.py -f init.json
```
## Useage
The genetic algorithm will output a list of statistics about each generation such as:
  - ```i```: Generation number
  - ```best```: Best fitness of an individual from the population
  - ```median```: Median fitness of an individual from the population
  - ```worst```: Worst fitness of an individual from the population
  - ```best pol count```: Number of polygons in the best 

The following is an example output:
```txt
i = 0  best = 0.8220410784313725  median = 0.804221274509804  worst = 0.7107117647058824  best pol count = 7
i = 1  best = 0.8289066666666667  median = 0.8147878431372549  worst = 0.8124129411764706  best pol count = 13
i = 2  best = 0.8308400980392157  median = 0.8228033333333333  worst = 0.8212820588235294  best pol count = 14
i = 3  best = 0.8353575490196078  median = 0.8218156862745098  worst = 0.8211895098039216  best pol count = 15
i = 4  best = 0.837801568627451  median = 0.8306085294117647  worst = 0.8279966666666667  best pol count = 14
i = 5  best = 0.8409974509803921  median = 0.8359427450980392  worst = 0.8334750980392157  best pol count = 15
```

Additionally, genetic algorithms will run for the specified number of ```generations```, once 
this has completed, a directory will be created where an image of the best individual 
from each population for every 10 generations will be saved. 

Using these images one can create videos or GIFS than can look like the following:
<p align="center">
  <img src="https://github.com/matous-elphick/2034-image-evolutionary-algorithms/blob/main/gifs/Offspring%20Generation%20VA%20-%2010000%20gens.gif" width="200" height="200" />
 </p>

Furthermore, the genetic algorithm will produce a plot of the median fitness of every generation
which will show the increase or decreases of fitness for every genertation. For example the following:
<p align="center">
  <img src="https://github.com/matous-elphick/2034-image-evolutionary-algorithms/blob/main/plots/myplot.png" width="400" height="300" />
</p>
