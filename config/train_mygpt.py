out_dir = 'out-europat'
eval_interval = 250 # keep frequent because we'll overfit
eval_iters = 200
log_interval = 10 # don't print too too often

always_save_checkpoint = False

wandb_log = False # override via command line if you like
wandb_project = 'machine-translation'
wandb_run_name = 'mygpt'

dataset = 'EuroPat'
gradient_accumulation_steps = 1
batch_size = 12
block_size = 640

init_from = 'mygpt'
n_layer = 6
n_head = 8
n_embd = 512
dropout = 0.0

learning_rate = 1e-3 # with baby networks can afford to go a bit higher
max_iters = 5000
lr_decay_iters = 5000 # make equal to max_iters usually
min_lr = 1e-4 # learning_rate / 10 usually
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small

warmup_iters = 100 # not super necessary potentially

device = 'cpu'  # run on cpu only
compile = False
