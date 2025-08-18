import importlib
from simple_parsing import ArgumentParser
from initializer import initialize_framework
from torch.utils.tensorboard import SummaryWriter
import random
import numpy as np
import torch

# Get arguments
parser = ArgumentParser()
parser.add_argument('--params', type=str, default='1st_order_2D', help='')
parser.add_argument('--results-base-directory', type=str, default='./', help='')
parser.add_argument('--encoder-only-epochs', type=int, default=1000, help='Number of iterations to train encoder only')
args = parser.parse_args()

# Import parameters
Params = getattr(importlib.import_module('params.' + args.params), 'Params')
params = Params(args.results_base_directory)
params.results_path += params.selected_primitives_ids + '/'

# Add encoder_only_epochs to params if not already there
if not hasattr(params, 'encoder_only_epochs'):
    params.encoder_only_epochs = args.encoder_only_epochs

# Initialize training objects
learner, evaluator, _ = initialize_framework(params, args.params)

# Start tensorboard writer
log_name = params.name+'_' + args.params + '_' + params.selected_primitives_ids
writer = SummaryWriter(log_dir='results/tensorboard_runs/' + log_name)
learner.traj_generator() # demo traj generation

# Train
for iteration in range(params.max_iterations + 1):
    # Update learning phase based on iteration
    
    # Evaluate model
    if iteration % params.evaluation_interval == 0 :
        metrics_acc, metrics_stab = evaluator.run(iteration=iteration)

        if params.save_evaluation:
            evaluator.save_progress(params.results_path, iteration, learner.model, writer)

        print('Metrics sum:', metrics_acc['metrics sum'], '; Number of unsuccessful trajectories:', metrics_stab['n spurious'])

    # Training step
    loss, loss_list, losses_names = learner.train_step()

    # Print progress with training phase info
    if iteration % 10 == 0:
        
        print(f"{iteration} Total cost: {loss.item()}")

    # Log losses in tensorboard
    for j in range(len(losses_names)):
        writer.add_scalar('losses/' + losses_names[j], loss_list[j], iteration)
    
    # Log training phase
    training_phase = 0 if iteration < params.encoder_only_epochs else 1  # 0: encoder only, 1: decoder only
    writer.add_scalar('training/phase', training_phase, iteration)

