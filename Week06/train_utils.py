import sys
import torch
import numpy as np
from utils import to_var, detach


def train_epoch(epoch, model, trn_ids, 
                criterion, optimizer, scheduler, 
                num_epochs, batch_size, seq_length):
    
    model.train()
    scheduler.step()
    states = model.init_hidden(batch_size)
    num_batches = trn_ids.size(1) // seq_length    
    trn_loss = 0.0
    trn_acc = 0.0
    
    for i in range(0, trn_ids.size(1) - seq_length, seq_length):
        inputs = to_var(trn_ids[:, i: i + seq_length])
        targets = to_var(trn_ids[:, (i + 1): (i + 1) + seq_length].contiguous())
                
        # Forward
        states = detach(states)
        outputs, states = model(inputs, states)
        
        # accuracy
        _, predictions = torch.max(outputs, dim=1)
        acc = torch.mean((predictions == targets.view(-1)).float())
        trn_acc = (trn_acc * i + acc.data[0]) / (i + 1) 
        
        # loss
        loss = criterion(outputs, targets.view(-1))
        trn_loss = (trn_loss * i + loss.data[0]) / (i + 1)  
        
        # backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), 0.3)
        optimizer.step()
        
        # report
        step = (i + 1) // seq_length
        sys.stdout.flush()
        sys.stdout.write('\rTraining: Epoch [%d/%d], Step [%d/%d], Loss: %.3f, Perp: %.2f, Acc: %-15.2f' % 
              (epoch + 1, num_epochs, step + 1, num_batches, trn_loss, np.exp(trn_loss), trn_acc))
    
    return trn_loss


def validate_epoch(epoch, model, val_ids, criterion, 
                   num_epochs, batch_size, seq_length):
    
    model.eval()
    states = model.init_hidden(batch_size)
    num_batches = val_ids.size(1) // seq_length        
    val_loss = 0.0
    val_acc = 0.0
    
    for i in range(0, val_ids.size(1) - seq_length, seq_length):
        inputs = to_var(val_ids[:, i: i + seq_length], volatile=True)
        targets = to_var(val_ids[:, (i + 1): (i + 1) + seq_length].contiguous())
                
        # Forward
        states = detach(states)
        outputs, states = model(inputs, states)
        
        # accuracy
        _, predictions = torch.max(outputs, dim=1)
        acc = torch.mean((predictions == targets.view(-1)).float())
        val_acc = (val_acc * i + acc.data[0]) / (i + 1) 
        
        # loss
        loss = criterion(outputs, targets.view(-1))
        val_loss = (val_loss * i + loss.data[0]) / (i + 1)  
                
        # report
        step = (i + 1) // seq_length
        sys.stdout.flush()
        sys.stdout.write('\rValidation: Epoch [%d/%d], Step [%d/%d], Loss: %.3f, Perp: %.2f, Acc: %-15.2f' % 
              (epoch + 1, num_epochs, step + 1, num_batches, val_loss, np.exp(val_loss), val_acc))
        
    return val_loss


def train(model, trn_ids, val_ids, 
          criterion, optimizer, scheduler, 
          num_epochs, batch_size, seq_length):
    
    best_loss = float('Inf')
    best_wgts = None
    
    trn_hist, val_hist = [], []

    for epoch in range(num_epochs):

        trn_loss = train_epoch(epoch, model, trn_ids, 
                               criterion, optimizer, scheduler, 
                               num_epochs, batch_size,  seq_length)
        
        val_loss = validate_epoch(epoch, model, val_ids, criterion, 
                                  num_epochs, batch_size, seq_length)
        
        trn_hist.append(trn_loss)
        val_hist.append(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            best_wgts = model.state_dict().copy()
            model.save(epoch, val_loss)
    
    # load best model weights
    model.load_state_dict(best_wgts)
    return trn_hist, val_hist