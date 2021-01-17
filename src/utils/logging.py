import os

import json

def write_params(model_dir, params):
    with open(os.path.join(model_dir, 'config.json'), 'w') as out:
        json.dump(params, out)