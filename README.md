![](http://www.cbs.dtu.dk/services/NetSurfP/)![maintained](http://img.shields.io/badge/status-maintained-red.png)<br>
![](http://www.cbs.dtu.dk/services/NetSurfP/)<br>
<I> The next revolution in biology will be by computational biologists</I>
# DecodET: The protein structure generator

## Problem statement
Protein sequencing is relatively much easier and cheaper to perform due to new technologies. But, with just a sequence we won't be able to determine the function of a protein. Protein structure determination is almost a necessary step in finding its function and even to engineer new proteins for varous applications. Several methods are currently used to determine the 3D structure of a protein, including X-ray crystallography, NMR spectroscopy, and Electron microscopy. They are extremely time consuming and expensive. Enter, Computational Biology and Machine learning. We need to build a Deep Learning model which takes a protein sequence (fasta format, see below) and gives a 3D structure (with coordinates of each amino acid in the protein, angles between the bonds, etc.
  
## BIO F111: A 5 minute crash course to get you up to speed
Amino acids are organic compounds that combine to form proteins. Amino acids and proteins are the building blocks of life. The 20 amino acids that are found within proteins convey a vast array of information. Each amino acid is represented as a single letter as shown below. 

<p align="center">
<img src= "https://i.pinimg.com/originals/57/fd/a8/57fda8cac0f5bfdabd2dfbe843ec93c2.png" alt="Amino acid codes" width=300>
</p><br>

Example of an protein sequence with 330 amino acids (This is called the fasta format, in the realm of Computational Biology): 
```
MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDWSTPSELGHAGLNGDILVWNPVLEDAFELSSMGIRVDADTLKHQLALTGDEDRLELEWHQALLRGEMPQTIGGGIGQSRLTMLLLQLPHIGQVQAGVWPAAVRESVPSLL
```

### Welcome to the 3D world

<p align="center"><img aligh="left" src= "https://cdn.kastatic.org/ka-perseus-images/71225d815cafcc09102504abdf4e10927283be98.png" alt="Protein Folding" width=300></p>

Several methods are currently used to determine the structure of a protein, including X-ray crystallography, NMR spectroscopy, and Electron microscopy. They are extremely time consuming and expensive. This is where Machine Learning and Deep Learning comes into the picture. Predicting 3D structure of protein from its amino acid sequence is one of the most important unsolved problems in biophysics and computational biology. Watch these interesting videos to gain more insight into the problem we are trying to solve: [TedX: The protein folding problem](https://www.youtube.com/watch?v=zm-3kovWpNQ&feature=youtu.be), [The protein folding revolution: Proteins and AI](https://youtu.be/cAJQbSLlonI), [BERTology: BERT meets biology](https://www.youtube.com/watch?v=q6Kyvy1zLwQ) and [Khanacademy: Protein basics](https://www.khanacademy.org/science/biology/macromolecules/proteins-and-amino-acids/a/orders-of-protein-structure).

## NetSurfP-2.0: The Baseline model
The [NetSurfP-2.0 paper](https://onlinelibrary.wiley.com/doi/abs/10.1002/prot.25674)'s model architecture is based on Bi-directional LSTM. NetSurfP-2.0 implements the blossom matrix for feature extraction from the sequence of amino acids. But recent progress in the field of Natural Language Processing like Google's BERT has opened up better ways to extract features. 

## DecodETv0.1

Task is to classify sections a protein sequence into 3 secondary structures:
- H = 4-turn helix (α helix). Minimum length 4 residues.
- E = extended strand in parallel and/or anti-parallel β-sheet conformation. Min length 2 residues.
- C = coil (residues which are not in any of the above conformations).

Example (Corresponding result for the above sequence): 
```
CCCCHHHHHHHHHHHHHHHHHHHHHHHCEEECCCCCEEECCCCCCCCCCCCCCCCEECCCCCCCCCEEECCCCCCHHHHHHHHCCCCCCCEEEEEEEEECCCCCCCCCCCCCEEEEEEEEEECCCCCCCHHHHHHHHHHHHHHHHHHHHHHHHHCCCCCCCCCCCEEEEHHHHHHHCCCCCHHHHHHHHHHHHCEEEEECCCCCCCCCCCCCCCCCCCECCCCECCCCCECCEEEEEEEECCCCEEEEEEEEEEECCHHHHHHHHHHHCCCCHHHCHHHHHHHCCCCCCEEEEEEEHHHHHHHHHCCCCHHHCCCCCCCHHHHHHCCCCC
```

### Inspiration from previous work:

[BERTology Meets Biology](https://arxiv.org/abs/2006.15222), [NetSurfP-2.0](https://onlinelibrary.wiley.com/doi/abs/10.1002/prot.25674) and [Evaluating Protein Transfer Learning with TAPE](https://arxiv.org/abs/1906.08230) have implemented Transformers and Bi-Directional LSTMs for the classification task. For v1, we will be taking a lot of inspiration from these papers and come up with a combination of them. We will be taking the [NetSurfP-2.0](https://onlinelibrary.wiley.com/doi/abs/10.1002/prot.25674) as our baseline.


You have the choice to help improve our model and help fix issues. We would rather have you work on novel models (not limiting to NLP) and compete with our model! We will be updating pending work, issues on our model soon. Meanwhile, it is up to you to find more papers and models. Novel models are most welcome. 

  
##  Dataset

We are borrowing the dataset from DTU Bioinformatics Institute's [NetSurfP-2.0](http://www.cbs.dtu.dk/services/NetSurfP/). The training set is represented this way: Raw data is given in Numpy (Python) compressed files with an array of pdb/chain ids (pdbids) and a 3-dimensional array (of shape ```(10848, 1632, 68)```) of input and output features. First dimension is protein samples, second dimension is sequence position and third dimension is input features. There are ```10848``` different protein sequences and largest sequence is ```1632``` amino acids long. And, each amino acid has the following data:

```
 [0:20] Amino Acids (sparse encoding)
 Unknown residues are stored as an all-zero vector
 [20:50] hmm profile
 [50] Seq mask (1 = seq, 0 = empty)
 [51] Disordered mask (0 = disordered, 1 = ordered)
 [52] Evaluation mask (For CB513 dataset, 1 = eval, 0 = ignore)
 [53] ASA (isolated)
 [54] ASA (complexed)
 [55] RSA (isolated)
 [56] RSA (complexed)
 [57:65] Q8 GHIBESTC (Q8 -> Q3: HHHEECCC)  ## We are replacing the different types of helices, Beta sheets from the OG dataset with just 3 classes - Helix, Beta sheet and Coil 
 [65:67] Phi+Psi
 [67] ASA_max
```

We have prepared a starter kit on [Colab](https://colab.research.google.com/drive/1Q0hEDSUQ9fFytPF6qWxGzfLkC-BJQv1I?usp=sharing#scrollTo=LNBsiy8e0Ci9&uniqifier=1) to get started quickly with the dataset. 
