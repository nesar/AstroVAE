# AstroVAE

Data reduction, emulation and inference using a combination of GP processes and Variational autoencoder. Currently applied for CMB angular power spectra C_l and matter power spectra P(k). In principle, this can be extended to image emulations as well. 

Parameter inference is done using MCMC for cosmological parameters, with public PLANCK/WMAP/SPT data.  

# Sync commands

## From phoenix -> laptop (AstroVAE/Cl_data/Data)
scp phoenix:/homes/nramachandra/AstroVAE/Cl_data/Model/*7500* mcs:/homes/nramachandra/DataP5/Model/
scp phoenix:/homes/nramachandra/AstroVAE/Cl_data/Data/norm*7500* mcs:/homes/nramachandra/DataP5/
scp phoenix:/homes/nramachandra/AstroVAE/Cl_data/Data/mean*7500* mcs:/homes/nramachandra/DataP5/
scp phoenix:/homes/nramachandra/AstroVAE/Cl_data/Data/encoded*7500* mcs:/homes/nramachandra/DataP5/


## From laptop -> phoenix
scp P*25.* mcs:/homes/nramachandra/DataP5/raw/


# Future implementations 
1. Error propoagation using Bayesian neural networks
1. t-SNE for reduction and visualization
2. RNN for time analysis
3. Adversarial AEs
