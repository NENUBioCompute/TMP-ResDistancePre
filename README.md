# TMP-SurResD
A deep-learning framework, TMP-SurResD (Transmembrane Proteins Surface Residue Distance Prediction), for simultaneously predicting relative distance of functional surface residues based on the combination of co-evolutionary information.

## The workflow and architecture of TMP-SurResD
![image](https://user-images.githubusercontent.com/52032167/193661439-5bb9cf68-bf08-4041-9fa6-5f5a1330722b.png)

## Details of the TMP-SurResD framework
![image](https://user-images.githubusercontent.com/52032167/193661583-79c9fefc-9775-4157-88ce-598ae5654acd.png)

## Download data
We provide the test dataset used in this study, you can download TEST.fasta to evaluate our method.

## Quick Start
### Requirements
Python 3.6
Pytorch
HH-suite for generating HHblits files (with the file suffix of .hhm)
deepMSA for generatinng MSAs
CCmpred for generating CCM

### Download TMP-SS
git clone https://github.com/NENUBioCompute/TMP-ResDistancePre.git

### Test & Evaluate in Command Line
We provide run.py that is able to run pre-trained models. Run it with:

python run.py -f sample/sample.fasta -p sample/hhblits/ -o results/
To set the path of fasta file, use --fasta or -f.
To set the path of generated HHblits files, use --hhblits_path or -p.
To save outputs to a directory, use --output or -o.

## Progress
README for running TMP-SurResD.
