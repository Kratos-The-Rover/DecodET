![](http://www.cbs.dtu.dk/services/NetSurfP/)![maintained](http://img.shields.io/badge/status-maintained-red.png)<br>
![](http://www.cbs.dtu.dk/services/NetSurfP/)<br>
<I> The next revolution in biology will be by computational biologists</I>
# DecodET: The protein structure predictor

 ## PROBLEM STATEMENT
 Implement a model which would provide good accuracy for converting the amino acid sequence into their respective secondary structures.
  
## BIO F111: A 5 minute crash course to get you up to speed

Amino acids are organic compounds that combine to form proteins. Amino acids and proteins are the building blocks of life. The 20 amino acids that are found within proteins convey a vast array of information. Each amino acid is represented as a single letter as shown below. 

<p align="center">
<img src= "https://i.pinimg.com/originals/57/fd/a8/57fda8cac0f5bfdabd2dfbe843ec93c2.png" alt="Amino acid codes" width=300>
</p><br>

Example of an protein sequence with 330 amino acids: 
```MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDWSTPSELGHAGLNGDILVWNPVLEDAFELSSMGIRVDADTLKHQLALTGDEDRLELEWHQALLRGEMPQTIGGGIGQSRLTMLLLQLPHIGQVQAGVWPAAVRESVPSLL```

### Welcome to the 3D world

<p align="center"><img aligh="left" src= "https://cdn.kastatic.org/ka-perseus-images/71225d815cafcc09102504abdf4e10927283be98.png" alt="Protein Folding" width=300></p>

Predicting 3D structure of protein from its amino acid sequence is one of the most important unsolved problems in biophysics and computational biology. 

<a href="https://www.youtube.com/watch?v=zm-3kovWpNQ&feature=youtu.be">The protein folding problem Tedx</a><br>
<a href="https://youtu.be/cAJQbSLlonI">The protein folding revolution: Proteins and AI</a><br>
<a href="https://www.youtube.com/watch?v=q6Kyvy1zLwQ">BERTology: BERT meets biology</a><br>
<a href="https://www.khanacademy.org/science/biology/macromolecules/proteins-and-amino-acids/a/orders-of-protein-structure">Khanacademy</a><br>

The Anfinsen experiment, showing that the structural characteristics of a protein are encoded in its primary sequence alone, is more than 50 years old.1 As a practical application of it, several methods have been developed over the last decades to predict from sequence only several protein structural features, including solvent accessibility, 2–8 secondary structure, backbone geometry, and disorder.<br>

<br>

##  DESCRIPTION
<hr>

So for the current project we aim to ouput the secondary structure of the protein depending on the amino acid sequence.So for the problem we are currently using the dataset from the DTU bioinformatics institute<br>
<a href="http://www.cbs.dtu.dk/services/NetSurfP/">Train data</a>
the model architecutre was based on the bi directional LSTM and for more information do read the paper
<a href="https://onlinelibrary.wiley.com/doi/abs/10.1002/prot.25674">NETSURF2.0</a>
Morever the NETSURF implements the blossom matrix for feature extraction from the sequence of amino acids.But recent progress in the field of Natural Language Processing has enabled us to get the same results using BERT Models.A recent 2019 paper using BERT models to get the encoding for each amino acids and it was shown to be similar to the BLOSSOM matrix 
<a href="https://arxiv.org/abs/2006.15222">BERTOLOGY</a>
Hence the pre trained data for creating the sentence encoding for each amino acid is provided in the TAPE paper and we use the pretrained BERT model for creating the encoding of the amino acid sequence and implement a model for classifying it into the secondary structures 
<a href="https://arxiv.org/abs/1906.08230">TAPE</a>
The 3 main secondary structures used for classification are :
<li> H = 4-turn helix (α helix). Minimum length 4 residues.
<li> E = extended strand in parallel and/or anti-parallel β-sheet conformation. Min length 2 residues.
<li> C = coil (residues which are not in any of the above conformations).
  

 
## IMPLEMENTATION

<hr>

The raw data format was downloaded for training and it consists of data.npy and pdbids.npy ,data.npy has the one for encoding for the amino acids and the secondary structures.<br>
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
 [57:65] Q8 GHIBESTC (Q8 -> Q3: HHHEECCC)
 [65:67] Phi+Psi
 [67] ASA_max
```
The code for converting it into the amino acid FASTA format along with the secondary structure has been provided in a collab file.<br>
The input for the bert encoding of the amino acids is basically a FASTA FORMAT
```
MGAEEEDTAILYPFTISGNDRNGNFTINFKGTPNSTNNGCIGYSYNGDWEKIEWEGSCDGNGNLVVEVPMSKIPAGVTSGEIQIWWHSGDLKMTDYKALEHHHHHH
```
The code for encoding is 
```
$ pip install tape_proteins
```

```
import torch
from tape import ProteinBertModel, TAPETokenizer
model = ProteinBertModel.from_pretrained('bert-base')
tokenizer = TAPETokenizer(vocab='iupac')  # iupac is the vocab for TAPE models, use unirep for the UniRep model

# Pfam Family: Hexapep, Clan: CL0536
sequence = 'GCTVEDRCLIGMGAILLNGCVIGSGSLVAAGALITQ'
token_ids = torch.tensor([tokenizer.encode(sequence)])
output = model(token_ids)
sequence_output = output[0]
pooled_output = output[1]

# NOTE: pooled_output is *not* trained for the transformer, do not use
# w/o fine-tuning. A better option for now is to simply take a mean of
# the sequence output

```
and it returns a 768 vectore which should then be passes into a model classifier which has 3 classes H,E and C
it should predict the secondary structure for each of the amino acids.For example the secondary structure for each of the amino acids given above are
```
CCCCCCCCCEEEEEECCCCCCCCEEEEEEEECCCCEEEEEEEEEECCEEEEEEEEEECCCCCEEEEEEEHHHCCCCCCEEEEEEEEECCCEEEEEEEECCEEEECC
```
<hr>
<I>MAY THE BEST MODEL WIN <I>




## Git setup cheatcode

```

git init
git add <file>
git commit -m "message"
git remote add origin <link or ssh>
git push --set-upstream origin master

```
