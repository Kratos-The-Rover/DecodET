
![](https://github.com/songlab-cal/tape/workflows/Build/badge.svg)
![](http://www.cbs.dtu.dk/services/NetSurfP/)
# DecodET
<hr>

## setting up git

```

git init
git add <file>
git commit -m "message"
git remote add origin <link or ssh>
git push --set-upstream origin master

```
## BIO F111 
The Anfinsen experiment, showing that the structural characteristics of a protein are encoded in its primary sequence alone, is more than 50 years old.1 As a practical application of it, several methods have been developed over the last decades to predict from sequence only several protein structural features, including solvent accessibility, 2–7secondary structure, backbone geometry, and disorder.<br>
Amino acids play central roles both as building blocks of proteins and as intermediates in metabolism. The 20 amino acids that are found within proteins convey a vast array of chemical versatility. Tertiary Structure of a proteinThe precise amino acid content, and the sequence of those amino acids, of a specific protein, is determined by the sequence of the bases in the gene that encodes that protein.<br>
Each amino acid is encoded as a single letter which can be referred from the image given
<img src= "https://www.google.com/url?sa=i&url=https%3A%2F%2Ftinycards.duolingo.com%2Fdecks%2F5uSD2spk%2Famino-acid-single-letter-code&psig=AOvVaw34Q9ivrVA-k9e4lnARqYMy&ust=1594405424657000&source=images&cd=vfe&ved=0CAIQjRxqFwoTCKDTnZrlwOoCFQAAAAAdAAAAABAD">

##  DESCRIPTION
<hr>

The protein structure prediction problem is the problem of determining the native structure of a protein, given its sequence of amino acids.<br>
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
  
 ## PROBLEM STATEMENT
  <hr>
 
 implement a model which would provide good accuracy for converting the amino acid sequence into their respective secondary structures.
 
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
