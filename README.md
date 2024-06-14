# D4C molecular property prediction
![License](https://img.shields.io/badge/license-D4C-red.svg)
![Version](https://img.shields.io/badge/version-1.0.0-brightgreen.svg)

This project is a deep learning application designed to predict molecular properties. The models implemented in this project feature interpretable and hierarchical architectures, including conventional graph convolutional models.

## How to start
1. Place the CSV file for training in the "_Data" folder.
    - The SMILES strings of molecules should be in the "compound" column.
    - There needs to be at least one molecular property for each corresponding molecule.
2. Check and choose the ID of the deep learning model in "network_refer.yaml".
3. Run the "main.py" file as

```sh
python main.py -n [network_id] -d [data_file_name] -t [property_column]
```
For example,
```sh
python main.py -n GCN -d Aqsoldb -t Solubility
```
The graph cache file will be saved in "_Graph" folder.
The trained result will be saved in "_Model" folder.

## Code Attribution and Licensing Information
This project includes code from the GC-GNN by Adem Rosenkvist Nielsen Aouichaoui (arnaou@kt.dtu.dk), licensed under the MIT License. 
https://github.com/gsi-lab/GC-GNN/tree/main 