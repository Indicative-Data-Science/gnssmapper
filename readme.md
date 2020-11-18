#**GNSS Simulation**

To run the simulation, create the experiment parameters as a yaml file (see template.yaml)
and execute run.py as a module with yaml filepath as an argument e.g 
python3 -m run data/configs/my_experiment.yaml 

--- 
#### Observation Emulation
There are four types of experiments:

 1) Random point simulation within a square
 ring with a specified width and ring thickness. 
 2) Random walk for a given time interval.
 3) N/E/S/W split of the square ring. Square thickness and 
 width needs to specified along with the desired orientation.
 4) Random walk for a chosen start time and given sampling frequency.

--- 
#### Map Algorithm








