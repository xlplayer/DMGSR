dataset = 'Tmall'
num_node = 43098
dim =256
epoch = 10
batch_size = 512
lr = 0.001
lr_dc = 0.1
lr_dc_step = 3
num_heads = 8
feat_drop = 0.15
weight_decay = 1e-3
lb_smooth = 0.4

density = 3
lambda_ = 0.1
validation = True

if dataset == "diginetica":
    lb_smooth = 0.4

elif dataset == "gowalla":
    lb_smooth = 0.8
    
elif dataset == "lastfm":
    lb_smooth = 0.8

elif dataset == "Tmall":
    num_node = 40728
    lb_smooth = 0.6
