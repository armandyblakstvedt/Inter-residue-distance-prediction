# Inter-residue distance prediction

This repository contains the code for a Convolutional Neural Network (CNN) that predicts the distance between pairs of residues in a protein sequence. The CNN is trained on a dataset of protein structures from the [Protein Data Bank](https://www.rcsb.org/). The dataset is preprocessed to extract the coordinates of the alpha-carbon atoms of each residue in the protein structures. The CNN is trained to predict the distance between pairs of residues based on the difference in their coordinates.

## Requirements

The code is written in Python and requires the following libraries:

- [NumPy](https://numpy.org/)
- [PyTorch](https://pytorch.org/)
- [scikit-learn](https://scikit-learn.org/)
- [matplotlib](https://matplotlib.org/)
- [pandas](https://pandas.pydata.org/)
- [biopython](https://biopython.org/)

## Data

TO get the data, run the following command:

```bash
./batch_download.sh -f protein-ids.txt -o ./data -p
```

To unzip the data, run the following command:

```bash
gunzip *.pdb.gz
```

The following filters are applied to the dataset:

- Structure Determination Methodology: Experimental
- Scientific Name of the source organism: Homo sapiens
- Experimental Method: X-Ray Diffraction
- Polymer entity type: Protein
- Refinement Resolution: 1.5 to 2
- Polymer length 300 - 400
