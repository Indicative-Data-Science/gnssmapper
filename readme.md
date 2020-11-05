#**GNSS Simulation**

To run the simulation, fill in the experiment parameters
of config.yaml and run orchestration.py. 

--- 
#### Observation Emulation
There are four types of experiments:

 1) Random point simulation within a square
 ring with a specified width and ring thickness. 
 2) Random walk for a given time interval.
 3) N/E/S/W split of the square ring. Square thickness and 
 width needs to specified along with the desired orientation.
 4) Random walk for a chosen start time and given sampling frequency.
 
The gen_receiver_points method in the orchestration script
will return a dictionary of receiver points and corresponding
GNSS observations for each experiment type. 

--- 
#### Map Algorithm








