# MatrixMethodNeuralNeutwork

data.py : This script runs over the ATLAS data, calculates several quantities and save into a json file.  <br />
signal.py : This script runs over the signal MC data, calculates several quantities and save into a json file.\\
uncertainty.py : This script runs over the background MC data, calculates the closure uncertainties for each of the 4 eta bin and save into a json file.\\
tightanalysis.py : This is the main script. Takes the data from the json files, and calculates the photon identification efficiencies, as well as tries to calculate tight-ID efficiencies.
