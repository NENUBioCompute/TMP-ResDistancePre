# TMP-ResDistancePre
### SurResD is a deep learning model based on the combination of Squeeze-and-Excitation(SE) block and Residual Block to predict the distance of residues on the surface of transmembrane protein.  
### TMResD is a deep learning model based on residual network, which is used to predict residue distance in transmembrane region of transmembrane protein.  
#### Both networks use feature which is a combination of HHM and CCM as input. HHM and CCM are calculated by HHBilts tool and CCMpred algorithm respectively. The output is the L×L residual-distance matrix.
#### If you want to use the model, follow this process:
#### (1) Prepare fasta files to be predicted
#### (2) Use the HHBlits tool to obtain the. HMM file: http://toolkit.genzentrum.lmu.de/hhblits/; The background database download link is   http://wwwuser.gwdg.de/~compbiol/data/hhsuite/databases/hhsuite_dbs/
#### (3) CCM was obtained using CCMPred (https://github.com/soedinglab/CCMpred), but before that you need to get multiple sequence alignment (MSA). The tool used in this article is deepMSA (https://zhanggroup.org/DeepMSA/), and you can also use a locally installed version to do the calculations faster. 
#### (4) Finally, the path in the code needs to be modified accordingly.

#### If you have any problem in using these models, please send me an email at chenqf830@nenu.edu.cn. Good luck for you! 
