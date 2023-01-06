# TMP-SurResD
A deep-learning framework, TMP-SurResD (Transmembrane Proteins Surface Residue Distance Prediction), for simultaneously predicting relative distance of functional surface residues based on the combination of co-evolutionary information.

## The workflow and architecture of TMP-SurResD
![image](https://user-images.githubusercontent.com/52032167/193661439-5bb9cf68-bf08-4041-9fa6-5f5a1330722b.png)

## Details of the TMP-SurResD framework
![image](https://user-images.githubusercontent.com/52032167/193661583-79c9fefc-9775-4157-88ce-598ae5654acd.png)

## Download data
We provide all the transmembrane protein sequences used in the manuscript, which are available in the './fasta/' directory.

## Quick Start
Here we provide the three trained models described in the manuscript, namely "24_ccmpred_90", "24_hhm_ccmpred_90", and "24_onehot_hhm_ccmpred90". The input dimension of model "24_ccmpred_90" is L×L×1 (CCM feature), "24_hhm_ccmpred_90" is L×L×61 (onehot+HHM), and "24_onehot_hhm_ccmpred90" is L×L×101 (onehot+HHM+CCM). The test data provided by us can be used to verify our proposed method.

### Requirements
Tools used in this study can be publicly available online:  PDBTM (http://pdbtm.enzim.hu);  Biopython (https://biopython.org/);  

CD-HIT (http://weizhong-lab.ucsd.edu/cd-hits/); \

TMP-SSurface-2.0 (https://github.com/NENUBioCompute/TMP-SSurface-2.0);\

HHblits (http://toolkit.genzentrum.lmu.de/hhblits/);\

Pytorch (https://pytorch.org/); \
DeepMSA (https://seq2fun.dcmb.med.umich.edu/DeepMSA/); \

Python 3.6 (https://www.python.org/); \
CCMpred (https://bitbucket.org/soedinglab/ccmpred);\
PSI-BLAST (ftp://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/LATEST/).\

### Download TMP-SSurface2, which is used to predict transmembrane protein surface residues.
```
git clone https://github.com/NENUBioCompute/TMP-SSurface-2.0.git
```

### Test & Evaluate in Command Line
We provide test.py that is able to run pre-trained models. Run it with:
```python
python test.py 
```

It is important to note that the 'test.py' provided is only to verify experimental results recorded in the manuscript and cannot be used directly to predict unknown protein sequences. If you need to predict an unknown protein sequence, the input data needs to be prepared in advance. Refer to the data feature extraction process shown below:

![hhhhhh](https://user-images.githubusercontent.com/52032167/211035424-1892cc72-4c0f-42d3-bee8-90647df254ad.png)

1. If you want to use the "24_ccmpred_90" model, just prepare the CCM features run out with CCMpred in advance. Then select three lines of code in the test.py file.
```python
x_test, y_test = con.main('ccmpred')
model_path = './model/24_ccmpred_90'
n = 1
```

2. If you want to use the "24_hhm_ccmpred_90" model, just prepare the CCM features run out with CCMpred and the HHM features generated with HHBlits in advance. Then select three lines of code in the test.py file.
```python
x_test, y_test = con.main('ccmpred')
model_path = './model/24_hhm_ccmpred_90'
n = 61
```
3. If you want to use the "24_onehot_hhm_ccmpred90" model, just prepare the CCM features run out of CCMpred, the HHM features generated by HHBlits, and the onehot encoding in advance. Then select three lines of code in the test.py file.
```python
x_test, y_test = con.main('ccmpred')
model_path = './model/24_onehot_hhm_ccmpred90'
n = 101
```

## Progress
README for running TMP-SurResD.
