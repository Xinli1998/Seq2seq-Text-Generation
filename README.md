
# Seq2seq-Text-Generation
# SCAN




#GROUP WORK


This is a re-implementation of the paper _Generalization without Systematicity: On the Compositional Skills of Sequence-to-Sequence Recurrent Networks_ (SCAN) by Lake and Baroni http://proceedings.mlr.press/v80/lake18a/lake18a.pdf

## Setup

Start by fetching the repository and checking out the submodule to get the dataset

```bash
git clone git@github.com:Xinli1998/SCAN.git
cd SCAN
git submodule update --init --recursive
```

Then make sure you have pytorch and tqdm installed into your environment. The project should work both on CPU and GPU.

## Training

To run a full training of the best overall model from the paper simply run the following (the default parameters should correspond to those used in the paper).

```bash
python src/scan/train.py \
    --train data/SCAN/simple_split/tasks_train_simple.txt \
    --valid data/SCAN/simple_split/tasks_test_simple.txt \
    --model lstm
```

Scripts for running all experiments using top-performing architectures and the over-best one are under `./scripts`.

## Statistics

After finishing running the experiments, run

```bash
cd stats
python get_stats.py
```

This will output a report with the results from the experiments and save some figures to the stats folder.




#INDIVIDUAL WORK on Transformer 

The main T5 implementation is in `individual/individual.py`.
Parameters:
'exp' refers to experiment 1 or 2 with value as "exp1" or "exp2"
'path_file' refers to data file
'iterations' means number of iterations


Firt, prepare environment:
```bash
nvidia-smi
pip install transformers
pip install pytorch_lightning
pip install sentencepiece datasets seqeval
```

Second, run on experiment1:
Data file : 'data/SCAN/simple_split/size_variations'
```bash
python individual/individual.py \
       --exp exp1 \
       --path_file data/SCAN/simple_split/size_variations/ \
       --iterations 8
```

Third, run on experiment2:
Data file : 'data/SCAN/length_split'
```bash
python individual/individual.py \
       --exp exp2 \
       --path_file data/SCAN/length_split/ \
       --iterations 8
```

