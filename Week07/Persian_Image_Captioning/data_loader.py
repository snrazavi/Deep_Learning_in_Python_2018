import os
import re
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from coco import COCO

import spacy
NLP = spacy.load('en')


def tokenizer(text):
    text = re.sub(b'\u200c'.decode("utf-8", "strict"), " ", text)   # replace half-spaces with spaces
    text = re.sub('\n', ' ', text)
    text = re.sub('-', ' - ', text)
    text = re.sub('[ ]+', ' ', text)
    text = re.sub('\.', ' .', text)
    text = re.sub('\طŒ', ' طŒ', text)
    text = re.sub('\ط›', ' ط›', text)
    text = re.sub('\طں', ' طں', text)
    text = re.sub('\. \. \.', '...', text)
    
    return [w.text for w in NLP.tokenizer(str(text))]


class CocoDataset(Dataset):
    """COCO Custom Dataset compatible with torch.utils.data.DataLoader."""
    def __init__(self, root, json, vocab, transform=None):
        """Set the path for images, captions and vocabulary wrapper.
        
        Args:
            root: image directory.
            json: coco annotation file path.
            vocab: vocabulary wrapper.
            transform: image transformer.
        """
        self.root = root
        self.coco = COCO(json)
        self.ids = list(self.coco.anns.keys())
        self.vocab = vocab
        self.transform = transform

    def __getitem__(self, index):
        """Returns one data pair (image and caption)."""
        ann_id = self.ids[index]
        img_id = self.coco.anns[ann_id]['image_id']
        caption = self.coco.anns[ann_id]['caption']
        path = self.coco.loadImgs(img_id)[0]['file_name']

        # Load image from disk and perform required transformations
        image = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        # numericalize: convert caption to token ids.
        tokens = tokenizer(str(caption))  # nltk.tokenize.word_tokenize(str(caption).lower())
        caption = []
        caption.append(self.vocab('<BOS>'))
        caption.extend([self.vocab(token) for token in tokens])
        caption.append(self.vocab('<EOS>'))
        target = torch.Tensor(caption)
        return image, target

    def __len__(self):
        return len(self.ids)


def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, caption).
    
    We should build custom collate_fn rather than using default collate_fn, 
    because merging caption (including padding) is not supported in default.

    Args:
        data: list of tuple (image, caption). 
            - image: torch tensor of shape (3, 224, 224).
            - caption: torch tensor of shape (?); variable length.

    Returns:
        images: torch tensor of shape (batch_size, 3, 224, 224).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length (descending order).
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions = zip(*data)

    # Merge images (from tuple of 3D tensor to 4D tensor).
    images = torch.stack(images, 0)

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]        
    return images, targets, lengths


def get_loader(root, json, vocab, transform, batch_size, shuffle, num_workers):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    # COCO caption dataset
    coco = CocoDataset(root=root,
                       json=json,
                       vocab=vocab,
                       transform=transform)
    
    # Data loader for COCO dataset
    # This will return (images, captions, lengths) for every iteration.
    # images: tensor of shape (batch_size, 3, 224, 224).
    # captions: tensor of shape (batch_size, padded_length).
    # lengths: list indicating valid length for each caption. length is (batch_size).
    data_loader = DataLoader(dataset=coco, 
                             batch_size=batch_size,
                             shuffle=shuffle,
                             num_workers=num_workers,
                             collate_fn=collate_fn)
    return data_loader
