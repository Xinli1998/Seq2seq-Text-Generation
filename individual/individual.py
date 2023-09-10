
from sklearn.model_selection import train_test_split
import sys


import argparse
import random
import os

import tqdm

import torch
from torch.utils.data import DataLoader

import torch
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np


import argparse
import glob
import os
import json
import time
import logging
import random
import re
from itertools import chain
from string import punctuation

# import nltk
# # nltk.download('punkt')
# from nltk.tokenize import sent_tokenize

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


from transformers import (
    AdamW,
    MT5ForConditionalGeneration,
    T5ForConditionalGeneration,
    T5Tokenizer,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    get_linear_schedule_with_warmup
)

def set_seed(seed):
                  random.seed(seed)
                  np.random.seed(seed)
                  torch.manual_seed(seed)
                  if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)

set_seed(42)

class WikiAnnDataset(Dataset):
    def __init__(self, tokenizer, path, max_len_inp=16, max_len_out=267):

        self.data_path = path
        self.max_len_input = max_len_inp
        self.max_len_output = max_len_out
        self.tokenizer = tokenizer
        self.inputs = []
        self.targets = []
        self._build()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        source_ids = self.inputs[index]["input_ids"].squeeze()
        target_ids = self.targets[index]["input_ids"].squeeze()

        src_mask    = self.inputs[index]["attention_mask"].squeeze()  # might need to squeeze
        target_mask = self.targets[index]["attention_mask"].squeeze()  # might need to squeeze

        return {"source_ids": source_ids, "source_mask": src_mask, "target_ids": target_ids, "target_mask": target_mask}

    def parse_action_command(line):
        command, action = line.strip().split(" OUT: ")
        command = command[3:]  # Remove "IN: "
        return command, action
  
    def _build(self):
        with open(self.data_path) as infile:
            for line in infile.readlines():
                command, action = line.strip().split(" OUT: ")
                command = command[4:] 
                # action = action.replace('I_', '') 
                # action = action.replace('_', '')
                # action = action.replace('TURN', '')

                input_ = "command: "+command 
                target = "action: "+action
                # target = target.lower()

                # tokenize inputs
                tokenized_inputs = self.tokenizer.batch_encode_plus(
                    [input_], max_length=self.max_len_input, padding='max_length', truncation=True,return_tensors="pt"
                )
                # tokenize targets
                tokenized_targets = self.tokenizer.batch_encode_plus(
                    [target],max_length=self.max_len_output, padding='max_length',truncation=True, return_tensors="pt"
                )

                self.inputs.append(tokenized_inputs)
                self.targets.append(tokenized_targets)
                
                
import pdb
class T5FineTuner(pl.LightningModule):
    def __init__(self, hparam):
        super(T5FineTuner, self).__init__()
        self.hparam = hparam
        self.model = T5ForConditionalGeneration.from_pretrained(
            hparam.model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            hparam.model_name_or_path)
        
        # self.tokenizer = T5Tokenizer.from_pretrained(
        #     hparam.model_name_or_path
        # )
        self.save_hyperparameters()
        self.collection = []

    def is_logger(self):
        return True

    def forward(
        self, input_ids, attention_mask=None, decoder_input_ids=None, decoder_attention_mask=None, lm_labels=None
    ):
        return self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=lm_labels,
        )

    def _step(self, batch):
        lm_labels = batch["target_ids"]
        lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100
        # pdb.set_trace()

        outputs = self(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            lm_labels=lm_labels,
            decoder_attention_mask=batch['target_mask']
        )

        loss = outputs[0]
        # print(loss)
        

        return loss

    def training_step(self, batch, batch_idx):
        
        loss = self._step(batch)
        # loss.backward()
        self.collection.append(loss.cpu().detach().numpy())

        tensorboard_logs = {"train_loss": loss}
        self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True)
        return {"loss": loss, "log": tensorboard_logs}

    def training_epoch_end(self, outputs):
        avg_train_loss = torch.stack([x["loss"] for x in outputs]).mean()
        tensorboard_logs = {"avg_train_loss": avg_train_loss}
        # print(f""avg_train_loss)

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch)
        tensorboard_logs = {"val_loss": loss}
        self.log('val_loss',loss, on_step=True, on_epoch=True, logger=True)
        
        # print(f'Current Validation Loss {loss}')
        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        tensorboard_logs = {"avg_val_loss": avg_loss}
        print(f'Last Five Validation Loss {torch.stack([x["val_loss"] for x in outputs[-5:]])}')
        

        # print(avg_loss)

    def configure_optimizers(self):
        "Prepare optimizer and schedule (linear warmup and decay)"

        model = self.model
        # no_decay = ["bias", "LayerNorm.weight"]
        # optimizer_grouped_parameters = [
        #     {
        #         "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
        #         "weight_decay": self.hparam.weight_decay,
        #     },
        #     {
        #         "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
        #         "weight_decay": 0.0,
        #     },
        # ]
        optimizer = AdamW(self.parameters(),
                          lr=self.hparam.learning_rate, eps=self.hparam.adam_epsilon)
        #,weight_decay=self.hparam.weight_decay
        
        self.opt = optimizer
        # return optimizer
        return [optimizer]

#     def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx,
#                    optimizer_closure, on_tpu, using_native_amp, using_lbfgs
#                        ):

#         optimizer.step(closure=optimizer_closure)
#         optimizer.zero_grad()
#         self.lr_scheduler.step()

    def get_tqdm_dict(self):
        tqdm_dict = {"loss": "{:.3f}".format(
            self.trainer.avg_loss), "lr": self.lr_scheduler.get_last_lr()[-1]}


        return tqdm_dict

    def train_dataloader(self):
        train_dataset = get_dataset(
            tokenizer=self.tokenizer,path=self.hparam.train)
        dataloader = DataLoader(train_dataset, batch_size=self.hparam.train_batch_size,
                                drop_last=True, shuffle=True, num_workers=12)
        # t_total = (
        #     (len(dataloader.dataset) //
        #      (self.hparam.train_batch_size * max(1, self.hparam.n_gpu)))
        #     // self.hparam.gradient_accumulation_steps
        #     * float(self.hparam.num_train_epochs)
        # )
        # scheduler = get_linear_schedule_with_warmup(
        #     self.opt, num_warmup_steps=self.hparam.warmup_steps, num_training_steps=t_total
        # )
        # self.lr_scheduler = scheduler
        
        # scheduler =torch.optim.lr_scheduler.LinearLR(self.opt)
        # self.lr_scheduler = scheduler
        return dataloader

    def val_dataloader(self):
        val_dataset = get_dataset(
            tokenizer=self.tokenizer, path=self.hparam.valid)
        return DataLoader(val_dataset, batch_size=self.hparam.eval_batch_size, num_workers=12)
    
    
    
    
def get_dataset(tokenizer, path):
    tokenizer.max_length = args.max_seq_length
    tokenizer.model_max_length = args.max_seq_length
    # dataset = load_dataset(args.data_dir, "en")
    return WikiAnnDataset(tokenizer=tokenizer,  path=path)



import textwrap
from tqdm import tqdm
import numpy as np
import pdb
from collections import defaultdict
def acc_process(acc_dict):
    stats = {}
    for k, v in acc_dict.items():
        acc = sum(v) / len(v)
        stats[k] = acc
    return stats

def eval(model,valid,tokenizer):

    accuracy_stats = {
    "command_length": defaultdict(list),
    "action_length": defaultdict(list)
    }
    input_dataset = WikiAnnDataset(tokenizer=tokenizer,  path=valid)
    # torch.device('cuda')
    dataloader = DataLoader(input_dataset, batch_size=1, num_workers=24,shuffle=True)
    model.model.eval()
    model = model.to("cuda")
    # dataloader.to('cuda')
    outputs = []
    targets = []
    texts = []
    t = 0
    r=0
    t_occ =0
    for batch in dataloader:
        # batch.to('cuda')
        max_len = np.argwhere(batch['target_ids'][0].numpy()==1)[0][0]
        outs = model.model.generate(input_ids=batch['source_ids'].cuda(),
                                    attention_mask=batch['source_mask'].cuda(),decoder_attention_mask=batch['target_mask'].cuda(),
                max_length=267,
                # min_length=max_len+1,
                do_sample=True,
                # max_length=50, 
                # top_p=0.95,
                # top_k=30,
                num_beams=10,
                num_return_sequences=1,
                # repetition_penalty=2.5, 
                # length_penalty=1.0, 
                early_stopping=True
                )
        # occ_outs = model.model.generate(input_ids=batch['source_ids'].cuda(),
        #                             attention_mask=batch['source_mask'].cuda(),decoder_attention_mask=batch['target_mask'].cuda(),
        #         max_length=max_len+1,
        #         # min_length=max_len+1,
        #         do_sample=True,
        #         # max_length=50, 
        #         # top_p=0.95,
        #         # top_k=30,
        #         num_beams=10,
        #         num_return_sequences=1,
        #         # repetition_penalty=2.5, 
        #         # length_penalty=1.0, 
        #         early_stopping=True
        #         )
        # pdb.set_trace()
        dec = [tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip().split() for ids in outs]
        # occ_dec = [tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip().split() for ids in occ_outs]
        target = [tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip().split()
                    for ids in batch["target_ids"]]
        text = [tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()
                    for ids in batch["source_ids"]]
        for i in range(len(target)):
            if dec[i] == target[i]:
                t += 1
                # pdb.set_trace()
                accuracy_stats['action_length'][len(target[0])-1].append(1)
                accuracy_stats['command_length'][len(text[0].split())-1].append(1)
            else:
                accuracy_stats['action_length'][len(target[0])-1].append(0)
                accuracy_stats['command_length'][len(text[0].split())-1].append(0)
                
            # if occ_dec[i] == target[i]:
            #     t_occ += 1
                
        r += 1
        if r% 1000 == 0:
            
            lines = textwrap.wrap("text:\n%s\n" % text[-1], width=100)
            print(f"Epoch: {r} -- Accurate: {t/r+1} -- Action_acc: {acc_process(accuracy_stats['action_length'])} -- Command_acc: {acc_process(accuracy_stats['command_length'])}")
            # print(f"Epoch: {r} -- Accurate: {t}-- ACC: {t/(r+1)}")
            print("\n".join(lines))
            print("True Action: %s" % target[-1])
            print("Predicted Action: %s" % dec[-1])
            # print("OCC_Predicted Action: %s" % occ_dec[-1])
            print("=====================================================================\n")
        # 
        
        
    accuracy_stats["command_length"] = acc_process(accuracy_stats["command_length"])
    accuracy_stats["action_length"] = acc_process(accuracy_stats["action_length"])
    accuracy_stats["accuracy"] = t/len(input_dataset)
    # accuracy_stats["oracle_accuracy"] = t_occ/len(input_dataset)
    print(accuracy_stats)  
    return accuracy_stats


def train_t5(train_file,valid_file,args,iterations):
    acc_list = []
    loss = []
    for e in range(iterations):


        model = T5FineTuner(args)
        model.model.train()
        model.to("cuda")

        checkpoint_callback = pl.callbacks.ModelCheckpoint(
          filename=args.output_dir+"{epoch}-{val_loss:.2f}", monitor="val_loss", mode="min", save_top_k=1
        )


        train_params = dict(
          # accumulate_grad_batches=args.gradient_accumulation_steps,
          gpus=args.n_gpu,
          max_epochs=args.num_train_epochs,
          #early_stop_callback=False,
          # precision= 32,
          # precision= 16 if args.fp_16 else 32,
          # amp_level=args.opt_level,
          # gradient_clip_val=args.max_grad_norm,
          # checkpoint_callback=checkpoint_callback,
          callbacks=[checkpoint_callback],
        )

        trainer = pl.Trainer(**train_params)
        trainer.fit(model)
        # loss.append(model.collection)
        acc = eval(model,valid_file,args.tokenizer)
        acc_list.append(acc)
    return acc_list,model

def funx_exp(acc_list,exp):
    if exp == "exp1":
        acc_total = []
        for i in range(len(acc_list)):
            acc_per = []
            for j in range(len(acc_list[i])):
                acc_per.append(acc_list_length[i][j]['accuracy'])
            acc_total.append(acc_per)
            # print(f"the average accuracy for experiment1 is {np.mean(acc_total)}")
            plot1(np.array(acc_total))
    else:    
        command_length = []
        action_length = []
        acc = []
        for i in range(len(acc_list_length)):
            command , action = acc_list_length[i]['command_length'],acc_list_length[i]['action_length']

            command = sorted(command.items(),key = lambda x:x[0])
            action = sorted(action.items(),key = lambda x:x[0])

            command = np.array([c[1] for c in command])
            command_length.append(command)

            action = np.array([c[1] for c in action])
            action_length.append(action)
            acc.append(acc_list_length[i]['accuracy'])
            print(f"the average accuracy for experiment2 is {np.mean(acc)}")
            plot2(np.array(action_length),np.array(command_length))
            
            
def plot1(acc_list):
    n,m = acc_list.shape
    label = np.array(["1%","2%","4%","8%","16%","32%","64%"])
    x_label = "Percent of Commands Used for Training"
    
    plt.bar(range(n),np.mean(acc_list,axis=1)*100,yerr = np.std(acc_list,axis=1)/np.sqrt(m)*100,tick_label=label)
    plt.ylim(0,110)
    plt.xlabel(x_label)
    plt.ylabel("Accuracy(%)")
    # plt.legend()
    for a,b in zip(np.array(range(n)),np.mean(acc_list,axis=1)*100):
                      if b==0:
                        continue
                      plt.text(a, b+2, '%.2f' % b + "%", ha='center', va= 'bottom',fontsize=9)
                        
def plot2(action_length,command_length):
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.bar(np.array(range(5)),np.mean(command_length,axis=0)*100,yerr = np.std(command_length,axis=0)/np.sqrt(5)*100,tick_label=np.array([4,6,7,8,9]),label='Ground-truth Command Sequence Length')
    ax1.set_ylim(0,100)
    for a,b in zip(np.array(range(5)),np.mean(command_length,axis=0)*100):
                      if b==0:
                        continue
                      ax1.text(a, b, '%.2f' % b + "%", ha='center', va= 'bottom',fontsize=10)
    ax1.set_xlabel("Command Length")
    ax1.set_ylabel("Accuracy(%)")

    ax2.bar(np.array(range(11)),np.mean(action_length,axis=0)*100,yerr = np.std(action_length,axis=0)/np.sqrt(5)*100,tick_label=np.array([24, 25, 26, 27, 28, 30, 32, 33, 36, 40, 48]),label='Ground-truth Action Sequence Length')
    ax2.set_ylim(0,100)
    for a,b in zip(np.array(range(11)),np.mean(action_length,axis=0)*100):
                      if b==0:
                        continue
                      ax2.text(a, b, '%.2f' % b + "%", ha='center', va= 'bottom',fontsize=10)
    ax2.set_xlabel("Action Length")
    ax2.set_ylabel("Accuracy(%)")
    plt.show()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, default="exp1")
    parser.add_argument("--path_file", type=str)
    parser.add_argument("--iterations", type=int,default=8)
    # parser.add_argument("--valid", type=str)
    # path_file = "cloud/"
    n = [1,2,4,8,16,32,64]


    args_0 = parser.parse_args()
    file = args_0.path_file
    if args_0.exp == 'exp1':
        acc_various = []
        for i in n:
            # if file.startswith("tasks_train_simple_p.txt"):

            train=file+"tasks_train_simple_p"+str(n[i])+".txt"
            valid=file+"tasks_test_simple_p"+str(n[i])+".txt"
            # valid = file + "test.txt"

            tokenizer = AutoTokenizer.from_pretrained("t5-base")
            args_dict = dict(
              # data_dir="wikiann", # path for data files
              output_dir="", # path to save the checkpoints
              model_name_or_path='t5-base',
              tokenizer_name_or_path='t5-base',
              tokenizer = tokenizer,
              max_seq_length=267,
              learning_rate=3e-4,
              # weight_decay=0.01,
              adam_epsilon=1e-8,
              # warmup_steps=0.05,
              train_batch_size=32,
              eval_batch_size=32,
              num_train_epochs=1,
              # gradient_accumulation_steps=1,
              n_gpu=1,
              # early_stop_callback=False,
              # fp_16=True, # if you want to enable 16-bit training then install apex and set this to true
              # opt_level='O1', # you can find out more on optimisation levels here https://nvidia.github.io/apex/amp.html#opt-levels-and-properties
              # max_grad_norm=0.5, # if you enable 16-bit training then set this to a sensible value, 0.5 is a good default
              seed=42,
          
              train = train,
              valid = valid
            )
            args = argparse.Namespace(**args_dict)


            acc_list,model = train_t5(train , valid ,args,args_0.iterations)
            acc_various.append(acc_list)
        funx_exp(acc_various,args_0.exp)
    else:

        train=file+"tasks_train_length.txt"
        valid=file+"tasks_test_length.txt"

        tokenizer = AutoTokenizer.from_pretrained("t5-base")
        args_dict = dict(
          # data_dir="wikiann", # path for data files
          output_dir="", # path to save the checkpoints
          model_name_or_path='t5-base',
          tokenizer_name_or_path='t5-base',
          tokenizer = tokenizer,
          max_seq_length=267,
          learning_rate=3e-4,
          # weight_decay=0.01,
          adam_epsilon=1e-8,
          # warmup_steps=0.05,
          train_batch_size=32,
          eval_batch_size=32,
          num_train_epochs=8,
          # gradient_accumulation_steps=1,
          n_gpu=1,
          # early_stop_callback=False,
          # fp_16=True, # if you want to enable 16-bit training then install apex and set this to true
          # opt_level='O1', # you can find out more on optimisation levels here https://nvidia.github.io/apex/amp.html#opt-levels-and-properties
          # max_grad_norm=0.5, # if you enable 16-bit training then set this to a sensible value, 0.5 is a good default
          seed=42,

          train = train,
          valid = valid
        )
        args = argparse.Namespace(**args_dict)


        acc_list,model = train_t5(train , valid ,args,args_0.iterations)
        funx_exp(acc_list,args_0.exp)




