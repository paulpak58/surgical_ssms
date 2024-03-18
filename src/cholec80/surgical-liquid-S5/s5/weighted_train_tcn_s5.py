import argparse
import pickle
from functools import partial
import wandb
import jax
import jax.numpy as np
from jax import random
from jax.scipy.linalg import block_diag
from jax import lax
#import torch
#import torch.nn as nn
import numpy as npy


from ssm_init import make_DPLR_HiPPO
from jax import random
from s5.ssm import init_S5SSM
from s5.bilinear_ssm import init_S5BilinearSSM
from s5.seq_model import RetrievalModel, BatchClassificationModel
from s5.train_helpers import create_train_state, reduce_lr_on_plateau,\
    linear_warmup, cosine_annealing, constant_lr


import mstcn
import torch



################################
# Extracts .pkl file of labels
################################
def get_data(data_path):
    with open(data_path, 'rb') as f:
        train_test_paths_labels = pickle.load(f)
    # train_paths_19 = train_test_paths_labels[0]
    train_paths_80 = train_test_paths_labels[0]
    val_paths_80 = train_test_paths_labels[1]
    # train_labels_19 = train_test_paths_labels[3]
    train_labels_80 = train_test_paths_labels[2]
    val_labels_80 = train_test_paths_labels[3]
    # train_num_each_19 = train_test_paths_labels[6]
    train_num_each_80 = train_test_paths_labels[4]
    val_num_each_80 = train_test_paths_labels[5]

    test_paths_80 = train_test_paths_labels[6]
    test_labels_80 = train_test_paths_labels[7]
    test_num_each_80 = train_test_paths_labels[8]

    print('train_paths_80  : {:6d}'.format(len(train_paths_80)))
    print('train_labels_80 : {:6d}'.format(len(train_labels_80)))
    print('valid_paths_80  : {:6d}'.format(len(val_paths_80)))
    print('valid_labels_80 : {:6d}'.format(len(val_labels_80)))
    # train_labels_80 = np.asarray(train_labels_80, dtype=np.int64)
    # val_labels_80 = np.asarray(val_labels_80, dtype=np.int64)
    # train_labels_80 = np.asarray(train_labels_80, dtype=np.int64)
    val_labels_80 = np.asarray(val_labels_80, dtype=np.int32)
    test_labels_80 = np.asarray(test_labels_80, dtype=np.int32)
    test_labels_80 = np.asarray(test_labels_80, dtype=np.int32)

    train_start_vidx = []
    count = 0
    for i in range(len(train_num_each_80)):
        train_start_vidx.append(count)
        count += train_num_each_80[i]

    val_start_vidx = []
    count = 0
    for i in range(len(val_num_each_80)):
        val_start_vidx.append(count)
        count += val_num_each_80[i]

    test_start_vidx = []
    count = 0
    for i in range(len(test_num_each_80)):
        test_start_vidx.append(count)
        count += test_num_each_80[i]

    return train_labels_80, train_num_each_80, train_start_vidx, val_labels_80, val_num_each_80, val_start_vidx,\
           test_labels_80, test_num_each_80, test_start_vidx



################################
# Extracts out spatial features
################################
def get_long_feature(start_index, lfb, LFB_length):
    long_feature = []
    long_feature_each = []
    for k in range(LFB_length):
        LFB_index = (start_index + k)
        LFB_index = int(LFB_index)
        long_feature_each.append(lfb[LFB_index])
    long_feature.append(long_feature_each)
    return long_feature


################################
# String to Boolean Helper
################################
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


################################
# Non-weighed, jax cross-entropy
################################
@partial(np.vectorize, signature="(c),()->()")
def cross_entropy_loss(logits, label):
    one_hot_label = jax.nn.one_hot(label, num_classes=logits.shape[0])
    return -np.sum(one_hot_label * logits)


################################
# Weighted, jax cross-entropy
################################
# @partial(np.vectorize, signature="(c),(),()->()")
def weighted_cross_entropy_loss(logits, label, weights):
    one_hot_label = jax.nn.one_hot(label, num_classes=logits.shape[-1])     # (batch, num_classes)
    weights = np.expand_dims(weights, axis=0)                               # (1, num_classes)
    return -np.sum(one_hot_label * logits * weights)


################################
# JAX vectorized accuracy
################################
@partial(np.vectorize, signature="(c),()->()")
def compute_accuracy(logits, label):
    return np.argmax(logits) == label



##################################  
# Learning rate and decay per step
##################################
def update_learning_rate_per_step(lr_params, state):
    decay_function, ssm_lr, lr, step, end_step, opt_config, lr_min = lr_params

    # Get decayed value
    lr_val = decay_function(step, lr, end_step, lr_min)
    ssm_lr_val = decay_function(step, ssm_lr, end_step, lr_min)
    step += 1

    # Update state
    state.opt_state.inner_states['regular'].inner_state.hyperparams['learning_rate'] = np.array(lr_val, dtype=np.float32)
    state.opt_state.inner_states['ssm'].inner_state.hyperparams['learning_rate'] = np.array(ssm_lr_val, dtype=np.float32)
    if opt_config in ["BandCdecay"]:
        # In this case we are applying the ssm learning rate to B, even though
        # we are also using weight decay on B
        state.opt_state.inner_states['none'].inner_state.hyperparams['learning_rate'] = np.array(ssm_lr_val, dtype=np.float32)

    return state, step



########################################
# Training step adapted with weighted-CE
########################################
@partial(jax.jit, static_argnums=(5, 6))
def train_step(state, rng, batch_inputs, batch_labels, batch_integration_timesteps, model, batchnorm, weighted_ce=True):
    ''' Performs single training step; adopted to go through entire video sequence '''
    def loss_fn(params):
        if batchnorm:

            # batch_inputs = np.reshape(batch_inputs, (np.shape(batch_inputs)[1], np.shape(batch_inputs)[2]))
            
            logits, mod_vars = model.apply(
                {"params": params, "batch_stats": state.batch_stats},
                batch_inputs, batch_integration_timesteps,
                rngs={"dropout": rng},
                mutable=["intermediates", "batch_stats"],
            )
        else:
            logits, mod_vars = model.apply(
                {"params": params},
                batch_inputs, batch_integration_timesteps,
                rngs={"dropout": rng},
                mutable=["intermediates"],
            )

        # batch_inputs = batch_inputs[0]
        logits = logits[0]                  # (1, T, C) -> (T, C)

        # Choelc80 Class Balance Weighting
        if weighted_ce:
            weights_train = np.asarray([
                1.6411019141231247,
                0.19090963801041133,
                1.0,
                0.2502662616859295,
                1.9176363911137977,
                0.9840248158200853,
                2.174635818337618
            ])
            loss = np.mean(weighted_cross_entropy_loss(logits, batch_labels, weights_train))
        else:
            loss = np.mean(cross_entropy_loss(logits, batch_labels))

        return loss, (mod_vars, logits)

    (loss, (mod_vars, logits)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    if batchnorm:
        state = state.apply_gradients(grads=grads, batch_stats=mod_vars["batch_stats"])
    else:
        state = state.apply_gradients(grads=grads)
    return state, loss


########################################
# Default eval step given
########################################
@partial(jax.jit, static_argnums=(4, 5))
def eval_step(batch_inputs, batch_labels, batch_integration_timesteps, state, model, batchnorm, weighted_ce=True):
    if batchnorm:
        logits = model.apply({"params": state.params, "batch_stats": state.batch_stats},
                             batch_inputs, batch_integration_timesteps,
                             )
    else:
        logits = model.apply({"params": state.params},
                             batch_inputs, batch_integration_timesteps,
                             )

    logits = logits[0]                  # (1, T, C) -> (T, C)

    # Choelc80 Class Balance Weighting
    if weighted_ce:
        weights_train = np.asarray([
            1.6411019141231247,
            0.19090963801041133,
            1.0,
            0.2502662616859295,
            1.9176363911137977,
            0.9840248158200853,
            2.174635818337618
        ])
        losses = weighted_cross_entropy_loss(logits, batch_labels, weights_train)
    else:
        losses = cross_entropy_loss(logits, batch_labels)


    accs = compute_accuracy(logits, batch_labels)
    return losses, accs, logits




#######################################################
# Training epoch with entire video sequence every batch
#######################################################
# def train_epoch(state, rng, model, trainloader, seq_len, in_dim, batchnorm, lr_params):
def train_epoch(
    state,
    rng,
    model,
    tcn_model,
    batchnorm,
    lr_params,\
    # Cholec80 path labels,
    train_labels_80,
    train_num_each_80,
    train_start_vidx,
    g_LFB_train,
):

    # Training, Val, Test Cholec80 Video Splits
    train_we_use_start_idx_80 = [x for x in range(40)]

    # Store metrics
    model = model(training=True)
    batch_losses = []

    decay_function, ssm_lr, lr, step, end_step, opt_config, lr_min = lr_params

    # for batch_idx, batch in enumerate(tqdm(trainloader)):

    for i in train_we_use_start_idx_80:
        # optimizer.zero_grad()
        # labels_phase = []
        labels_phase = np.array([])
        for j in range(train_start_vidx[i], train_start_vidx[i]+train_num_each_80[i]):
            # labels_phase.append(train_labels_80[j][0])
            labels_phase = np.append(labels_phase, train_labels_80[j][0])
        # labels_phase = torch.LongTensor(labels_phase)
        # labels_phase = torch.LongTensor(labels_phase).to(device)

        long_feature = get_long_feature(start_index=train_start_vidx[i],
                                        lfb=g_LFB_train, LFB_length=train_num_each_80[i])

        long_feature = torch.from_numpy(npy.array(long_feature)).float()
        video_fe = long_feature.transpose(2,1).cuda()                   # (1, 2048, seq_len)

        out_features = tcn_model.forward(video_fe)[-1]

        # out_features = out_features.squeeze(1)

        # Prepare data as jax inputs for liquid-s5
        inputs = np.array(out_features.detach().cpu().numpy()).astype(float)        # (1, 7, vid_len)

        # inputs = inputs[0]
        #labels = np.asarray(labels_phase.numpy()).astype(float)
        labels = labels_phase.astype(float)
        # inputs = np.reshape(inputs, (np.shape(inputs)[1], np.shape(inputs)[2]))
        # inputs = inputs[None,:]
        inputs = np.transpose(inputs, axes=(0,2,1))
        # inputs = np.expand_dims(inputs, axis=0)
        # print(f'input shape {np.shape(inputs)} ; labels shape {np.shape(labels)}')

        integration_times = None
        # inputs, labels, integration_times = prep_batch(batch, seq_len, in_dim)

        # Training step
        rng, drop_rng = jax.random.split(rng)
        state, loss = train_step(
            state,
            drop_rng,
            inputs,
            labels,
            integration_times,
            model,
            batchnorm,
        )
        batch_losses.append(loss)
        lr_params = (decay_function, ssm_lr, lr, step, end_step, opt_config, lr_min)
        state, step = update_learning_rate_per_step(lr_params, state)

    # Return average loss over batches
    return state, np.mean(np.array(batch_losses)), step


def validate(
    state,
    model,
    tcn_model,
    batchnorm,\
    # Cholec80 path labels,
    val_labels_80,
    val_num_each_80,
    val_start_vidx,
    g_LFB_val,
    step_rescale=1.0
):
    ''' Validation function to loop over batches '''
    val_we_use_start_idx_80 = [x for x in range(40)]
    # test_we_use_start_idx_80 = [x for x in range(32)]

    model = model(training=False, step_rescale=step_rescale)
    losses, accuracies, preds = np.array([]), np.array([]), np.array([])
    for i in val_we_use_start_idx_80:
        labels_phase = np.array([])
        for j in range(val_start_vidx[i], val_start_vidx[i]+val_num_each_80[i]):
            labels_phase = np.append(labels_phase, val_labels_80[j][0])
        long_feature = get_long_feature(start_index=val_start_vidx[i],
                                        lfb=g_LFB_val, LFB_length=val_num_each_80[i])

        long_feature = torch.from_numpy(npy.array(long_feature)).float()
        video_fe = long_feature.transpose(2,1).cuda()                   # (1, 2048, seq_len)

        out_features = tcn_model.forward(video_fe)[-1]
        #out_features = out_features.squeeze(1)

        # Prepare data as jax inputs for liquid-s5
        inputs = np.array(out_features.detach().cpu().numpy()).astype(float)        # (1, 7, vid_len)
        inputs = np.transpose(inputs, axes=(0,2,1))
        labels = labels_phase.astype(float)
        integration_times = None

        # Evaluation step
        loss, acc, pred = eval_step(inputs, labels, integration_times, state, model, batchnorm)
        losses = np.append(losses, loss)
        accuracies = np.append(accuracies, acc)
        preds = np.append(preds, np.argmax(pred, axis=1), axis=0)
    ave_loss, ave_accu = np.mean(losses), np.mean(accuracies)
    return ave_loss, ave_accu, preds



def train(args):

    # Retrieve labels and spatial features from pkl file    
    train_labels_80, train_num_each_80, train_start_vidx,\
        val_labels_80, val_num_each_80, val_start_vidx,\
        test_labels_80, test_num_each_80, test_start_vidx = get_data('../../train_val_paths_labels1.pkl')
    with open("../../LFB/g_LFB50_train0.pkl", 'rb') as f:
        g_LFB_train = pickle.load(f)
    with open("../../LFB/g_LFB50_val0.pkl", 'rb') as f:
        g_LFB_val = pickle.load(f)
    with open("../../LFB/g_LFB50_test0.pkl", 'rb') as f:
        g_LFB_test = pickle.load(f)
    print("Spatial features loaded!")
    print("g_LFB_train shape:", g_LFB_train.shape)
    print("g_LFB_val shape:", g_LFB_val.shape)
    print("g_LFB_test shape:", g_LFB_test.shape)


    args.USE_WANDB = True
    args.wand_project = 'liquids5'
    args.wandb_entity = 'exp0'

    # W&B logging setup
    #if args.USE_WANDB:
    #    wandb.init(project=args.wand_project, job_type='train', config=vars(args), entity=args.wandb_entity)
    #else:
    #    wandb.init(mode='offline')
    best_test_loss = 1e9
    best_test_acc = -1e9
    ssm_size = args.ssm_size_base
    ssm_lr = args.ssm_lr_base
    block_size = int(ssm_size/args.blocks)
    lr = args.lr_factor*ssm_lr
    #wandb.log({'block_size': block_size})


    # Random seed setup
    print("[*] Setting Randomness...")
    key = random.PRNGKey(args.jax_seed)
    init_rng, train_rng = random.split(key, num=2)


    # Configure Dataset specific parameters
    padded = False
    retrieval = False
    speech = False
    init_rng, key = random.split(init_rng, num=2)       # split init_rng for initializing state matrix A
    n_classes = 7                                       # # of phases
    train_size = 40                                     # 40 training videos
    #seq_len = train_num_each_80[0]                      # starting sequence length is length of first videoa
    seq_len = args.seq_len
    print(f'Initial sequence length {seq_len}')
    # seq_len = 768                                       # TODO: Check -- Arbitrary starting sequence?


    #in_dim = 2048                                       # feature dim output from the spatial extractor
    in_dim = 32                                       # feature dim output from the spatial extractor
    all_eval_preds = np.array([])                       # for storing predictions for all videos



    # Set up TCN backbone
    mstcn_causal_conv = True
    mstcn_layers = 8
    mstcn_f_maps = 32
    mstcn_f_dim= 2048
    out_features = 32
    mstcn_stages = 2
    tcn_model = mstcn.MultiStageModel(mstcn_stages, mstcn_layers, mstcn_f_maps, mstcn_f_dim, out_features, mstcn_causal_conv)

    tcn_model_path = "/ppak/surgical_ncp/src/cholec_pipeline/TCN_weights/"
    tcn_model_name = "TeCNO50_epoch_6_train_9935_val_8924_test_8603"
    ckpt = torch.load(tcn_model_path+tcn_model_name+'.pth')
    out_names = [
        'stages.0.conv_1x1.weight',
        'stages.0.conv_1x1.bias',
        'stages.0.conv_1x1.weight',
        'stages.0.conv_1x1.bias',
        'stages.0.conv_out_classes.weight',
        'stages.0.conv_out_classes.bias',
        'stage1.conv_out_classes.weight',
        'stage1.conv_out_classes.bias'
    ]
    new_ckpt = {}
    for k,v in ckpt.items():
        if k not in out_names:
            new_ckpt[k] = v
    rslt = tcn_model.load_state_dict(new_ckpt,strict=False)
    tcn_model.cuda()
    tcn_model.eval()


    print(f"[*] Starting S5 Training on `{args.dataset}` =>> Initializing...")

    # Initialize state matrix A using approximation to HiPPO-LegS matrix
    Lambda, _, B, V, B_orig = make_DPLR_HiPPO(block_size)

    if args.conj_sym:
        block_size = block_size // 2
        ssm_size = ssm_size // 2

    Lambda = Lambda[:block_size]
    V = V[:, :block_size]
    Vc = V.conj().T

    # If initializing state matrix A as block-diagonal, put HiPPO approximation
    # on each block
    Lambda = (Lambda * np.ones((args.blocks, block_size))).ravel()
    V = block_diag(*([V] * args.blocks))
    Vinv = block_diag(*([Vc] * args.blocks))

    print("Lambda.shape={}".format(Lambda.shape))
    print("V.shape={}".format(V.shape))
    print("Vinv.shape={}".format(Vinv.shape))

    if args.bilinear:
        ssm_init_fn = init_S5BilinearSSM(H=args.d_model,
                                         P=ssm_size,
                                         Lambda_re_init=Lambda.real,
                                         Lambda_im_init=Lambda.imag,
                                         V=V,
                                         Vinv=Vinv,
                                         C_init=args.C_init,
                                         discretization=args.discretization,
                                         dt_min=args.dt_min,
                                         dt_max=args.dt_max,
                                         conj_sym=args.conj_sym,
                                         clip_eigs=args.clip_eigs,
                                         bidirectional=args.bidirectional)

    else:
        ssm_init_fn = init_S5SSM(H=args.d_model,
                                 P=ssm_size,
                                 Lambda_re_init=Lambda.real,
                                 Lambda_im_init=Lambda.imag,
                                 V=V,
                                 Vinv=Vinv,
                                 C_init=args.C_init,
                                 discretization=args.discretization,
                                 dt_min=args.dt_min,
                                 dt_max=args.dt_max,
                                 conj_sym=args.conj_sym,
                                 clip_eigs=args.clip_eigs,
                                 bidirectional=args.bidirectional)


    if retrieval:
        # Use retrieval head for AAN task
        print("Using Retrieval head for {} task".format(args.dataset))
        model_cls = partial(
            RetrievalModel,
            ssm=ssm_init_fn,
            d_output=n_classes,
            d_model=args.d_model,
            n_layers=args.n_layers,
            padded=padded,
            activation=args.activation_fn,
            dropout=args.p_dropout,
            prenorm=args.prenorm,
            batchnorm=args.batchnorm,
            bn_momentum=args.bn_momentum,
        )

    else:
        model_cls = partial(
            BatchClassificationModel,
            ssm=ssm_init_fn,
            d_output=n_classes,
            d_model=args.d_model,
            n_layers=args.n_layers,
            padded=padded,
            activation=args.activation_fn,
            dropout=args.p_dropout,
            mode=args.mode,
            prenorm=args.prenorm,
            batchnorm=args.batchnorm,
            bn_momentum=args.bn_momentum,
        )
    
    state = create_train_state(model_cls,
                               init_rng,
                               padded,
                               retrieval,
                               in_dim=in_dim,
                               bsz=args.bsz,
                               seq_len=seq_len,
                               weight_decay=args.weight_decay,
                               batchnorm=args.batchnorm,
                               opt_config=args.opt_config,
                               ssm_lr=ssm_lr,
                               lr=lr,
                               dt_global=args.dt_global)

    # Training Loop over epochs
    best_loss, best_acc, best_epoch = 100000000, -100000000.0, 0  # This best loss is val_loss
    count, best_val_loss = 0, 100000000  # This line is for early stopping purposes
    lr_count, opt_acc = 0, -100000000.0  # This line is for learning rate decay
    step = 0  # for per step learning rate decay
    steps_per_epoch = int(train_size/args.bsz)

    # Combining test and validation
    test_labels_80 = np.concatenate((np.array(val_labels_80), np.array(test_labels_80)), axis=0)
    test_num_each_80 = np.concatenate((np.array(val_num_each_80), np.array(test_num_each_80)), axis=0)
    test_start_vidx = np.array(test_start_vidx) + np.array(val_num_each_80).sum()
    test_start_vidx = np.concatenate((np.array(val_start_vidx), np.array(test_start_vidx)), axis=0)
    g_LFB_test = np.concatenate((np.array(g_LFB_val), np.array(g_LFB_test)), axis=0)

    for epoch in range(args.epochs):
        print(f"[*] Starting Training Epoch {epoch + 1}...")

        # SCHEDULER set up
        if epoch < args.warmup_end:
            print("using linear warmup for epoch {}".format(epoch+1))
            decay_function = linear_warmup
            end_step = steps_per_epoch * args.warmup_end
        elif args.cosine_anneal:
            print("using cosine annealing for epoch {}".format(epoch+1))
            decay_function = cosine_annealing
            # for per step learning rate decay
            end_step = steps_per_epoch * args.epochs - (steps_per_epoch * args.warmup_end)
        else:
            print("using constant lr for epoch {}".format(epoch+1))
            decay_function = constant_lr
            end_step = None

        # TODO: Switch to letting Optax handle this.
        #  Passing this around to manually handle per step learning rate decay.
        lr_params = (decay_function, ssm_lr, lr, step, end_step, args.opt_config, args.lr_min)

        train_rng, skey = random.split(train_rng)
        state, train_loss, step = train_epoch(
            state,
            skey,
            model_cls,
            tcn_model,
            args.batchnorm,
            lr_params,
            train_labels_80,
            train_num_each_80,
            train_start_vidx,
            g_LFB_train,
        )


        # print(f"[*] Running Epoch {epoch + 1} Validation...")
        # val_loss, val_acc = validate(
        #     state,
        #     model_cls,
        #     args.batchnorm,
        #     val_labels_80=val_labels_80,
        #     val_num_each_80=val_num_each_80,
        #     val_start_vidx=val_start_vidx,
        #     g_LFB_val=g_LFB_val
        # )

        # Test pipeline runs from videos 40-80
        print(f"[*] Running Epoch {epoch + 1} Test...")
        test_loss, test_acc, test_preds = validate(
            state,
            model_cls,
            tcn_model,
            args.batchnorm,
            val_labels_80=test_labels_80,
            val_num_each_80=test_num_each_80,
            val_start_vidx=test_start_vidx,
            g_LFB_val=g_LFB_test,
        )
        val_loss, val_acc = test_loss, test_acc

        print(f"\n=>> Epoch {epoch + 1} Metrics ===")
        print(
            # f"\tTrain Loss: {train_loss:.5f} -- Val Loss: {val_loss:.5f} --Test Loss: {test_loss:.5f} --"
            f" Val Accuracy: {val_acc:.4f}"
            f" Test Accuracy: {test_acc:.4f}"
        )
        # For early stopping purposes
        if val_loss < best_val_loss:
            count = 0
            best_val_loss = val_loss
        else:
            count += 1

        if val_acc > best_acc:
            # Increment counters etc.
            count = 0
            best_loss, best_acc, best_epoch = val_loss, val_acc, epoch
            best_test_loss, best_test_acc = test_loss, test_acc

            # Save predictions for matlab eval later
            all_eval_preds = test_preds
            print(f'evaluation shape {all_eval_preds.shape}')

            # filename = "/home/ppak/S5/s5/liquidS5_weights.pt"
            filename = "../../liquidS5_weights.pt"
            #wandb.save(filename)
            print(f'Attempting model save to {filename}...')

        # For learning rate decay purposes:
        input = lr, ssm_lr, lr_count, val_acc, opt_acc
        lr, ssm_lr, lr_count, opt_acc = reduce_lr_on_plateau(input, factor=args.reduce_factor, patience=args.lr_patience, lr_min=args.lr_min)

        # Print best accuracy & loss so far...
        print(
            f"\tBest Val Loss: {best_loss:.5f} -- Best Val Accuracy:"
            f" {best_acc:.4f} at Epoch {best_epoch + 1}\n"
            f"\tBest Test Loss: {best_test_loss:.5f} -- Best Test Accuracy:"
            f" {best_test_acc:.4f} at Epoch {best_epoch + 1}\n"
        )
        #wandb.log(
        #    {
        #        "Training Loss": train_loss,
        #        "Val loss": val_loss,
        #        "Val Accuracy": val_acc,
        #        "Test Loss": test_loss,
        #        "Test Accuracy": test_acc,
        #        "count": count,
        #        "Learning rate count": lr_count,
        #        "Opt acc": opt_acc,
        #        "lr": state.opt_state.inner_states['regular'].inner_state.hyperparams['learning_rate'],
        #        "ssm_lr": state.opt_state.inner_states['ssm'].inner_state.hyperparams['learning_rate']
        #    }
        #)
        #wandb.run.summary["Best Val Loss"] = best_loss
        #wandb.run.summary["Best Val Accuracy"] = best_acc
        #wandb.run.summary["Best Epoch"] = best_epoch
        #wandb.run.summary["Best Test Loss"] = best_test_loss
        #wandb.run.summary["Best Test Accuracy"] = best_test_acc
        if count > args.early_stop_patience:
            break

    # Dump evaluations to pkl and npy file
    pred_name = args.eval_file_name
    pkl_name = pred_name + '.pkl'
    npy_name = pred_name + '.npy'
    with open(pkl_name, 'wb') as f:
        pickle.dump(np.array(all_eval_preds), f)
    with open(npy_name, 'wb') as f:
        np.save(f, np.array(all_eval_preds))





def main(args):
    raise NotImplementedError('main')



if __name__=='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--USE_WANDB", type=str2bool, default=False, help="whether to log with wandb")
    parser.add_argument("--wandb_project", type=str, default=None, help="wandb project name")
    parser.add_argument("--dir_name", type=str, default='./cache_dir', help="name of cache dir")
    parser.add_argument("--dataset", type=str, default='cholec80', help="dataset name")

    # Model Parameters
    parser.add_argument("--n_layers", type=int, default=6, help="Number of layers in the network")
    parser.add_argument("--d_model", type=int, default=128, help="Number of features, i.e. H, ""dimension of layer inputs/outputs")
    parser.add_argument("--ssm_size_base", type=int, default=256, help="SSM Latent size, i.e. P")
    parser.add_argument("--blocks", type=int, default=8, help="How many blocks, J, to initialize with")
    parser.add_argument("--C_init", type=str, default="trunc_standard_normal",
                        choices=["trunc_standard_normal", "lecun_normal", "complex_normal"],
                        help="Options for initialization of C: \\"
                             "trunc_standard_normal: sample from trunc. std. normal then multiply by V \\ " \
                             "lecun_normal sample from lecun normal, then multiply by V\\ " \
                             "complex_normal: sample directly from complex standard normal")
    parser.add_argument("--bilinear", type=str2bool, default=False, help="use bilinear LDS?")
    parser.add_argument("--discretization", type=str, default="zoh", choices=["zoh", "bilinear"])
    parser.add_argument("--mode", type=str, default="pool", choices=["pool", "last"],
                        help="options: (for classification tasks) \\" \
                             " pool: mean pooling \\" \
                             "last: take last element")
    parser.add_argument("--activation_fn", default="half_glu1", type=str, choices=["full_glu", "half_glu1", "half_glu2", "gelu"])
    parser.add_argument("--conj_sym", type=str2bool, default=True, help="whether to enforce conjugate symmetry")
    parser.add_argument("--clip_eigs", type=str2bool, default=False, help="whether to enforce the left-half plane condition")
    parser.add_argument("--bidirectional", type=str2bool, default=False, help="whether to use bidirectional model")
    parser.add_argument("--dt_min", type=float, default=0.001, help="min value to sample initial timescale params from")
    parser.add_argument("--dt_max", type=float, default=0.1, help="max value to sample initial timescale params from")

    # Optimization Parameters
    parser.add_argument("--prenorm", type=str2bool, default=True, help="True: use prenorm, False: use postnorm")
    parser.add_argument("--batchnorm", type=str2bool, default=True, help="True: use batchnorm, False: use layernorm")
    parser.add_argument("--bn_momentum", type=float, default=0.95, help="batchnorm momentum")
    parser.add_argument("--bsz", type=int, default=64, help="batch size")
    parser.add_argument("--epochs", type=int, default=100, help="max number of epochs")
    parser.add_argument("--early_stop_patience", type=int, default=1000, help="number of epochs to continue training when val loss plateaus")
    parser.add_argument("--ssm_lr_base", type=float, default=1e-3, help="initial ssm learning rate")
    parser.add_argument("--lr_factor", type=float, default=1, help="global learning rate = lr_factor*ssm_lr_base")
    parser.add_argument("--dt_global", type=str2bool, default=False, help="Treat timescale parameter as global parameter or SSM parameter")
    parser.add_argument("--lr_min", type=float, default=0, help="minimum learning rate")
    parser.add_argument("--cosine_anneal", type=str2bool, default=True, help="whether to use cosine annealing schedule")
    parser.add_argument("--warmup_end", type=int, default=1, help="epoch to end linear warmup")
    parser.add_argument("--lr_patience", type=int, default=1000000, help="patience before decaying learning rate for lr_decay_on_val_plateau")
    parser.add_argument("--reduce_factor", type=float, default=1.0, help="factor to decay learning rate for lr_decay_on_val_plateau")
    parser.add_argument("--p_dropout", type=float, default=0.0, help="probability of dropout")
    parser.add_argument("--weight_decay", type=float, default=0.05, help="weight decay value")
    parser.add_argument("--opt_config", type=str, default="standard", choices=['standard',
                                                                               'BandCdecay',
                                                                               'BfastandCdecay',
                                                                               'noBCdecay'],
                        help="Opt configurations: \\ " \
               "standard:       no weight decay on B (ssm lr), weight decay on C (global lr) \\" \
                 "BandCdecay:     weight decay on B (ssm lr), weight decay on C (global lr) \\" \
                 "BfastandCdecay: weight decay on B (global lr), weight decay on C (global lr) \\" \
                 "noBCdecay:      no weight decay on B (ssm lr), no weight decay on C (ssm lr) \\")
    parser.add_argument("--jax_seed", type=int, default=1919, help="seed randomness")
    parser.add_argument("--eval_file_name", type=str, default="eval.pkl", help="name of eval file", required=False)

    parser.add_argument('--seq_len', type=int, default=768, help='initial sequence length used for dummy input')
    train(parser.parse_args())

    
