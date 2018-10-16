import sys
import math
import torch
import torch.nn.functional as F

from utils import time_since, to_var


def seq2seq_loss(outputs, targets, criterion):
    out_len = outputs.size(0)
    tgt_len = targets.size(0)
    bs = outputs.size(1)
    
    if out_len < tgt_len:
        outputs = F.pad(outputs, (0, 0, 0, 0, 0, tgt_len - out_len))
    elif out_len > tgt_len:
        targets = F.pad(targets, (0, 0, 0, out_len - tgt_len))
        tgt_len = out_len
    
    loss = criterion(outputs.view(tgt_len * bs, -1), targets.view(-1))
    return loss


def report_stats(i, num_batches, epoch, num_epochs, loss):
    percent = int((i + 1) / num_batches * 100)

    sys.stdout.flush()
    sys.stdout.write(
        '\r({:3d}%): Epoch [{:2d}/{:2d}], Step [{:3d}/{:3d}], loss = {:.4f}\t\t'.format(
            percent, epoch + 1, num_epochs, i + 1, num_batches, loss))


def train_step(model, train_dl, optimizer, criterion, tfr, epoch, num_epochs):
    
    model.encoder.train()
    model.decoder.train()
    
    epoch_loss = 0
    
    for i, (src_var, tgt_var, lengths) in enumerate(train_dl):
        
        src_var, tgt_var = to_var(src_var), to_var(tgt_var)
        
        # forward step
        outputs, _ = model(src_var, lengths, tgt_var, tfr)
        loss = seq2seq_loss(outputs, tgt_var, criterion)
        epoch_loss = (epoch_loss * i + loss.data[0]) / (i + 1)
        report_stats(i, len(train_dl), epoch, num_epochs, epoch_loss)
        
        # backward step
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), 5.0)
        optimizer.step()
    
    print()
    return epoch_loss


def validate_step(model, valid_dl, criterion, epoch, num_epochs):
    
    model.encoder.eval()
    model.decoder.eval()
    
    epoch_loss = 0
    
    for i, (src_var, tgt_var, lengths) in enumerate(valid_dl):
        
        src_var, tgt_var = to_var(src_var, True), to_var(tgt_var, True)
        
        # forward step
        outputs, _ = model(src_var, lengths)
        loss = seq2seq_loss(outputs, tgt_var, criterion)
        epoch_loss = (epoch_loss * i + loss.data[0]) / (i + 1)
        report_stats(i, len(valid_dl), epoch, num_epochs, epoch_loss)
    
    print('\n') 
    return epoch_loss