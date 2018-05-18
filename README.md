# AstroVAE
Data reduction, emulation and inference using a combination of GP emulator and Variational autoencoder 

## Architecture: 

![Model](Old/ArchitectureFullAE.png "Full model")

3. t-SNE
4. RNN for time analysis


7. Adversarial AEs


# Sync commands

## From phoenix -> laptop (AstroVAE/Cl_data/Data)
scp phoenix:/homes/nramachandra/AstroVAE/Cl_data/Model/*7500* mcs:/homes/nramachandra/DataP5/Model/
scp phoenix:/homes/nramachandra/AstroVAE/Cl_data/Data/norm*7500* mcs:/homes/nramachandra/DataP5/
scp phoenix:/homes/nramachandra/AstroVAE/Cl_data/Data/mean*7500* mcs:/homes/nramachandra/DataP5/
scp phoenix:/homes/nramachandra/AstroVAE/Cl_data/Data/encoded*7500* mcs:/homes/nramachandra/DataP5/


## From laptop -> phoenix
scp P*25.* mcs:/homes/nramachandra/DataP5/raw/

