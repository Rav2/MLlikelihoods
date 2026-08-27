#!/bin/bash

for ii in $(seq 1 20);
do
	sbatch "likelihood_1908_08215-${ii}.sh"
done	
