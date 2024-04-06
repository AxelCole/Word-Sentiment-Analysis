from typing import List
import copy

from LoRA import LoRA

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from transformers import get_scheduler
from datasets import Dataset


import pandas as pd

class Classifier:
    """
    The Classifier: complete the definition of this class template by providing a constructor (i.e. the
    __init__() function) and the 2 methods train() and predict() below. Please do not change the signature
    of these methods
     """



    ############################################# complete the classifier class below
    
    def __init__(self):
        """
        This should create and initilize the model. Does not take any arguments.
        
        """
        # Define DistilBERT as our base model:
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self.model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=3)
        for i in range(len(self.model.distilbert.transformer.layer)):
            self.model.distilbert.transformer.layer[i].attention.q_lin = LoRA(self.model.distilbert.transformer.layer[i].attention.q_lin, 4)
            self.model.distilbert.transformer.layer[i].attention.v_lin = LoRA(self.model.distilbert.transformer.layer[i].attention.v_lin, 4)

        # Freeze all the pretrained weights
        # Frozen weights won't have their gradients computed during loss.backward()
        # Yet, the optimizer might still want to update them even with no available gradient
        # That's why we give only the non frozen weights to the optimizer next
        learned_params = []
        for name, param in self.model.named_parameters():
            if 'adapter' in name or 'classifier' in name:
                learned_params.append(param)
            else:
                param.requires_grad = False
        
        # Update only the non frozen weights
        self.optimizer = AdamW(learned_params, lr=5e-5)

    def tokenize_function(self, examples):
        return self.tokenizer(examples["text"], padding="max_length", truncation=True)
        
    def process(self, filename):
        df = pd.read_csv(filename, sep='\t', header=None)
        texts = ['We want to do sentiment analysis regarding '+df[1][i]+'. Focus on the word ### '+df[2][i]+' ### at position '+df[3][i]+' in this sentence : ### '+df[4][i]+' ###' for i in range(len(df))]
        my_dict = {"text": texts, "label": df[0].map(lambda x: (x=='neutral')+2*(x=='positive')).to_list()}
        dataset = Dataset.from_dict(my_dict)
        tokenized_datasets = dataset.map(self.tokenize_function, batched=True)
        tokenized_datasets = tokenized_datasets.remove_columns(["text"])
        tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
        tokenized_datasets.set_format("torch")
        return tokenized_datasets

    def train_one_epoch(self, train_dataloader, device, lr_scheduler=None):
        running_tloss = 0.
        if device == 'cpu':
            device_type = 'cpu'
        else:
            device_type = 'cuda'
        for i, batch in enumerate(train_dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            # Runs the forward pass with autocasting.
            with torch.autocast(device_type=device_type, dtype=torch.float16):
                outputs = self.model(**batch)
                loss = outputs.loss
            # Scales loss.  Calls backward() on scaled loss to create scaled gradients.
            # Backward passes under autocast are not recommended.
            # Backward ops run in the same dtype autocast chose for corresponding forward ops.
            self.scaler.scale(loss).backward()

            # scaler.step() first uncast the gradients to float32 and unscales the gradients of the optimizer's assigned params
            # If these gradients do not contain infs or NaNs, optimizer.step() is then called,
            # otherwise, optimizer.step() is skipped.
            self.scaler.step(self.optimizer)
            # Updates the scale for next iteration.
            self.scaler.update()

            if lr_scheduler != None:
                lr_scheduler.step()

            self.optimizer.zero_grad()

            running_tloss += loss.item()
        training_loss = running_tloss/(i+1)
        return training_loss
    
    
    def train(self, train_filename: str, dev_filename: str, device: torch.device):
        """
        Trains the classifier model on the training set stored in file trainfile
        PLEASE:
          - DO NOT CHANGE THE SIGNATURE OF THIS METHOD
          - PUT THE MODEL and DATA on the specified device! Do not use another device
          - DO NOT USE THE DEV DATA AS TRAINING EXAMPLES, YOU CAN USE THEM ONLY FOR THE OPTIMIZATION
         OF MODEL HYPERPARAMETERS
        """
        
        train_dataset = self.process(train_filename)
        eval_dataset = self.process(dev_filename)

        train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=8)
        eval_dataloader = DataLoader(eval_dataset, batch_size=8)

        self.model.to(device)

        num_epochs = 5
        num_training_steps = num_epochs * len(train_dataloader)
        lr_scheduler = get_scheduler(name="linear", optimizer=self.optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)
        self.scaler = torch.cuda.amp.GradScaler() # for mixed-precision training

        best_vloss = 100000.
        for epoch in range(num_epochs):
            self.model.train() # for dropout during training and not inference
            training_loss = self.train_one_epoch(train_dataloader, device, lr_scheduler)
            
            running_vloss = 0.
            self.model.eval()
            # Disable gradient computation and reduce memory consumption.
            with torch.no_grad():
                for i, batch in enumerate(eval_dataloader):
                    batch = {k: v.to(device) for k, v in batch.items()}
                    outputs = self.model(**batch)
                    running_vloss += outputs.loss
            validation_loss = running_vloss/(i+1)

            print('Training Loss : {} | Validation Loss : {}'.format(training_loss, validation_loss))
            if validation_loss < best_vloss:
                best_vloss = validation_loss
                best_model = copy.deepcopy(self.model)
        self.model = best_model


    def predict(self, data_filename: str, device: torch.device) -> List[str]:
        """Predicts class labels for the input instances in file 'datafile'
        Returns the list of predicted labels
        PLEASE:
          - DO NOT CHANGE THE SIGNATURE OF THIS METHOD
          - PUT THE MODEL and DATA on the specified device! Do not use another device
        """
        test_dataset = self.process(data_filename)
        test_dataloader = DataLoader(test_dataset, batch_size=8)
        running_test_loss = 0.
        self.model.to(device)
        self.model.eval()

        predictions = []
        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            for i, batch in enumerate(test_dataloader):
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = self.model(**batch)
                predictions += outputs.logits.argmax(dim=-1).tolist()
                running_test_loss += outputs.loss
        test_loss = running_test_loss/(i+1)
        print('Test Loss : {}'.format(test_loss))
        final_pred = []
        for i in range(len(predictions)):
            if predictions[i]==0:
                final_pred.append('negative')
            elif predictions[i]==1:
                final_pred.append('neutral')
            else:
                final_pred.append('positive')
        return final_pred
