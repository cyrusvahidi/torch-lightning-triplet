import torch

import os
import json

import torch.nn as nn

def load_model(model_dir, parallel=True):
    config = json.load(open(os.path.join(model_dir, 'config.json')))

    model_fpath = config['ckpt_path']
    
    model.load_state_dict(torch.load(model_fpath))
    model.eval()
    
    return model, config