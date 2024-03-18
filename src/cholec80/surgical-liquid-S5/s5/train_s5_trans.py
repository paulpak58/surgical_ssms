#some code adapted from https://github.com/YuemingJin/MTRCNet-CL
# and https://github.com/YuemingJin/TMRNet
import argparse
import torch
from torch import optim
from torch import nn
#import numpy as np
import numpy    
import pickle, time
import random
from sklearn import metrics
import copy
import mstcn
from dummy_transformer import Transformer2_3_1
import os, subprocess
from functools import partial

import jax
import jax.numpy as np
from jax.scipy.linalg import block_diag
from jax import lax

from ssm_init import make_DPLR_HiPPO
from s5.ssm import init_S5SSM
from s5.bilinear_ssm import init_S5BilinearSSM
from s5.seq_model import RetrievalModel, BatchClassificationModel
from s5.train_helpers import create_train_state, reduce_lr_on_plateau,\
    linear_warmup, cosine_annealing, constant_lr

import flax
from flax.training.checkpoints import save_checkpoint, orbax_utils, restore_checkpoint
from orbax.checkpoint import PyTreeCheckpointer, CheckpointManagerOptions, CheckpointManager



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
# jax vectorized accuracy
################################
@partial(np.vectorize, signature="(c),()->()")
def compute_accuracy(logits, label):
    return np.argmax(logits) == label


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


########################################
# Default eval step given
########################################
@partial(jax.jit, static_argnums=(4, 5))
def eval_step(batch_inputs, batch_labels, batch_integration_timesteps, state, model, batchnorm, weighted_ce=False):
    if batchnorm:
        logits = model.apply({"params": state.params, "batch_stats": state.batch_stats}, batch_inputs, batch_integration_timesteps)
    else:
        logits = model.apply({"params": state.params}, batch_inputs, batch_integration_timesteps)

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

    # print('train_paths_19  : {:6d}'.format(len(train_paths_19)))
    # print('train_labels_19 : {:6d}'.format(len(train_labels_19)))
    print('train_paths_80  : {:6d}'.format(len(train_paths_80)))
    print('train_labels_80 : {:6d}'.format(len(train_labels_80)))
    print('valid_paths_80  : {:6d}'.format(len(val_paths_80)))
    print('valid_labels_80 : {:6d}'.format(len(val_labels_80)))

    # train_labels_19 = np.asarray(train_labels_19, dtype=np.int64)
    #train_labels_80 = np.asarray(train_labels_80, dtype=np.float32)
    #val_labels_80 = np.asarray(val_labels_80, dtype=np.float32)
    #test_labels_80 = np.asarray(test_labels_80, dtype=np.float32)

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

def get_long_feature(start_index, lfb, LFB_length):
    long_feature = []
    long_feature_each = []
    # 上一个存在feature的index
    for k in range(LFB_length):
        LFB_index = (start_index + k)
        LFB_index = int(LFB_index)
        long_feature_each.append(lfb[LFB_index])
    long_feature.append(long_feature_each)
    return long_feature



class Transformer(nn.Module):
    def __init__(self, mstcn_f_maps, mstcn_f_dim, out_features, len_q, sequence_length=30):
        super(Transformer, self).__init__()
        self.num_f_maps = mstcn_f_maps  # 32
        self.dim = mstcn_f_dim  # 2048
        self.num_classes = out_features  # 7
        self.len_q = len_q

        self.transformer = Transformer2_3_1(d_model=out_features, d_ff=mstcn_f_maps, d_k=mstcn_f_maps,
                                        d_v=mstcn_f_maps, n_layers=1, n_heads=8, len_q = sequence_length)
        self.fc = nn.Linear(mstcn_f_dim, out_features, bias=False)


    def forward(self, x, long_feature):
        out_features = x.transpose(1,2)
        inputs = []
        for i in range(out_features.size(1)):
            if i<self.len_q-1:
                input = torch.zeros((1, self.len_q-1-i, self.num_classes)).cuda()
                input = torch.cat([input, out_features[:, 0:i+1]], dim=1)
            else:
                input = out_features[:, i-self.len_q+1:i+1]
            inputs.append(input)
        inputs = torch.stack(inputs, dim=0).squeeze(1)
        feas = torch.tanh(self.fc(long_feature).transpose(0,1))
        output = self.transformer(inputs, feas)
        #output = output.transpose(1,2)
        #output = self.fc(output)
        return output


def train_s5_trans(args):

    train_labels_80, train_num_each_80, train_start_vidx,\
        val_labels_80, val_num_each_80, val_start_vidx,\
        test_labels_80, test_num_each_80, test_start_vidx = get_data('../../train_val_paths_labels1.pkl')

    with open("../../LFB/g_LFB50_train0.pkl", 'rb') as f:
        g_LFB_train = pickle.load(f)
    with open("../../LFB/g_LFB50_val0.pkl", 'rb') as f:
        g_LFB_val = pickle.load(f)
    with open("../../LFB/g_LFB50_test0.pkl", 'rb') as f:
        g_LFB_test = pickle.load(f)
    print("load completed")
    print("g_LFB_train shape:", g_LFB_train.shape)
    print("g_LFB_val shape:", g_LFB_val.shape)

    out_features = 7
    batch_size = 1
    mstcn_causal_conv = True
    learning_rate = 1e-3
    #min_epochs = 12
    max_epochs = 25
    mstcn_layers = 8
    mstcn_f_maps = 32
    mstcn_f_dim= 2048
    mstcn_stages = 2
    sequence_length = 30
    seed = 1
    print("Random Seed: ", seed)
    random.seed(seed)
    #np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    use_gpu = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_gpu else "cpu")
    weights_train = np.asarray([1.6411019141231247,
                0.19090963801041133,
                1.0,
                0.2502662616859295,
                1.9176363911137977,
                0.9840248158200853,
                2.174635818337618,])
    #criterion_phase = nn.CrossEntropyLoss(weight=torch.from_numpy(weights_train).float().to(device))
    criterion_phase1 = nn.CrossEntropyLoss()

    # Model parameters
    ssm_size = args.ssm_size_base
    ssm_lr = args.ssm_lr_base
    block_size = int(ssm_size/args.blocks)
    lr = args.lr_factor*ssm_lr

    # Random seed setup
    print("[*] Setting Randomness...")
    key = jax.random.PRNGKey(args.jax_seed)
    init_rng, train_rng = jax.random.split(key, num=2)

    # Configure Dataset specific parameters
    padded = False
    retrieval = False
    speech = False
    init_rng, key = jax.random.split(init_rng, num=2)       # split init_rng for initializing state matrix A
    n_classes = 7                                       # # of phases
    train_size = 40                                     # 40 training videos
    seq_len = args.seq_len
    print(f'Initial sequence length {seq_len}')
    # seq_len = 768                                       # TODO: Check -- Arbitrary starting sequence?
    in_dim = 2048                                       # feature dim output from the spatial extractor
    all_eval_preds = np.array([])                       # for storing predictions for all videos



    
    # Initialize state matrix A using approximation to HiPPO-LegS matrix
    Lambda, _, B, V, B_orig = make_DPLR_HiPPO(block_size)
    if args.conj_sym:
        block_size = block_size // 2
        ssm_size = ssm_size // 2

    Lambda = Lambda[:block_size]
    V = V[:, :block_size]
    Vc = V.conj().T

    # If initializing state matrix A as block-diagonal, put HiPPO approximation on each block
    Lambda = (Lambda * np.ones((args.blocks, block_size))).ravel()
    V = block_diag(*([V] * args.blocks))
    Vinv = block_diag(*([Vc] * args.blocks))
    print("Lambda.shape={}".format(Lambda.shape))
    print("V.shape={}".format(V.shape))
    print("Vinv.shape={}".format(Vinv.shape))

    # Whether causal or not
    if args.bilinear:
        ssm_init_fn = init_S5BilinearSSM(
            H=args.d_model,
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
            bidirectional=args.bidirectional
        )

    else:
        ssm_init_fn = init_S5SSM(
            H=args.d_model,
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
            bidirectional=args.bidirectional
        )

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
    
    state = create_train_state(
        model_cls,
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
        dt_global=args.dt_global
    )

    # Load in stored checkpoint 
    restore_checkpoint(ckpt_dir=args.restore_path, target=state, step=1080)      # by default, takes last checkpoint state
    #new_state = restore_checkpoint(ckpt_dir=args.restore_path, target=None, step=None)      # by default, takes last checkpoint state
    #new_state = new_state['model']
    #state = flax.serialization.from_state_dict(target=state, state=new_state)

    
    # Evaluate S5
    model_cls = model_cls(training=False, step_rescale=1.0)


    # Initialzie TransSV_Net
    model1 = Transformer(mstcn_f_maps, mstcn_f_dim, out_features, sequence_length)
    model1 = model1.to(torch.float32).cuda()
    optimizer1 = optim.Adam(model1.parameters(), lr=learning_rate)
    best_model_wts = copy.deepcopy(model1.state_dict())
    best_val_accuracy_phase = 0.0
    correspond_train_acc_phase = 0.0
    best_epoch = 0

    train_we_use_start_idx_80 = [x for x in range(40)]
    val_we_use_start_idx_80 = [x for x in range(8)]
    test_we_use_start_idx_80 = [x for x in range(32)]
    for epoch in range(max_epochs):
        torch.cuda.empty_cache()
        random.shuffle(train_we_use_start_idx_80)
        train_idx_80 = []
        model1.train()
        train_loss_phase = 0.0
        train_corrects_phase = 0
        batch_progress = 0.0
        running_loss_phase = 0.0
        minibatch_correct_phase = 0.0
        train_start_time = time.time()
        for i in train_we_use_start_idx_80:
            #optimizer.zero_grad()
            optimizer1.zero_grad()
            labels_phase = np.array([])
            labels_phase_torch = []
            for j in range(train_start_vidx[i], train_start_vidx[i]+train_num_each_80[i]):
                labels_phase = np.append(labels_phase, np.copy(train_labels_80[j][0]))
                labels_phase_torch.append(train_labels_80[j][0])
            labels_phase_torch = torch.LongTensor(labels_phase_torch).to(device)

            long_feature = get_long_feature(start_index=train_start_vidx[i],
                                            lfb=g_LFB_train, LFB_length=train_num_each_80[i])

            # Prepare data in the foramt of S5
            video_fe = long_feature
            inputs = np.array(video_fe).astype(float)
            labels = labels_phase.astype(float)
            integration_times = None

            # Evaluation step
            loss, acc, pred = eval_step(inputs, labels, integration_times, state, model_cls, args.batchnorm)
            out_features = np.transpose(pred, axes=(1,0))
            out_features = torch.tensor(np.expand_dims(out_features, axis=0).tolist(), device=device).detach()
            #long_feature = torch.from_numpy(np.asarray(long_feature)).to(device)
            long_feature = torch.from_numpy(numpy.array(long_feature)).float().to(device)
            p_classes1 = model1(out_features, long_feature)

            #p_classes = y_classes.squeeze().transpose(1, 0)
            #clc_loss = criterion_phase(p_classes, labels_phase)
            p_classes1 = p_classes1.squeeze()

            # Compute loss 
            clc_loss = criterion_phase1(p_classes1, labels_phase_torch)

            _, preds_phase = torch.max(p_classes1.data, 1)

            loss = clc_loss
            #print(loss.data.cpu().numpy())
            loss.backward()
            #optimizer.step()
            optimizer1.step()

            running_loss_phase += clc_loss.data.item()
            train_loss_phase += clc_loss.data.item()

            batch_corrects_phase = torch.sum(preds_phase == labels_phase_torch.data)
            train_corrects_phase += batch_corrects_phase
            minibatch_correct_phase += batch_corrects_phase

            batch_progress += 1
            if batch_progress * batch_size >= len(train_we_use_start_idx_80):
                percent = 100.0
                print('Batch progress: %s [%d/%d]' % (str(percent) + '%', len(train_we_use_start_idx_80),
                                                    len(train_we_use_start_idx_80)), end='\n')
            else:
                percent = round(batch_progress * batch_size / len(train_we_use_start_idx_80) * 100, 2)
                print('Batch progress: %s [%d/%d]' % (
                    str(percent) + '%', batch_progress * batch_size, len(train_we_use_start_idx_80)), end='\r')

        train_elapsed_time = time.time() - train_start_time
        train_accuracy_phase = float(train_corrects_phase) / len(train_labels_80)
        train_average_loss_phase = train_loss_phase

        # Sets the module in evaluation mode.
        # model.eval()
        model1.eval()
        val_loss_phase = 0.0
        val_corrects_phase = 0
        val_start_time = time.time()
        val_progress = 0
        val_all_preds_phase = []
        val_all_labels_phase = []
        val_acc_each_video = []

        with torch.no_grad():
            for i in val_we_use_start_idx_80:
                labels_phase = np.array([])
                labels_phase_torch = []
                for j in range(val_start_vidx[i], val_start_vidx[i]+val_num_each_80[i]):
                    labels_phase = np.append(labels_phase, np.copy(val_labels_80[j][0]))
                    labels_phase_torch.append(val_labels_80[j][0])
                labels_phase_torch = torch.LongTensor(labels_phase_torch).to(device)

                long_feature = get_long_feature(start_index=val_start_vidx[i],
                                                lfb=g_LFB_val, LFB_length=val_num_each_80[i])

                # Prepare data in the foramt of S5
                video_fe = long_feature
                inputs = np.array(video_fe).astype(float)
                labels = labels_phase.astype(float)
                integration_times = None

                # Evaluation step
                loss, acc, pred = eval_step(inputs, labels, integration_times, state, model_cls, args.batchnorm)

                out_features = np.transpose(pred, axes=(1,0))
                out_features = torch.tensor(np.expand_dims(out_features, axis=0).tolist(), device=device).detach()
                long_feature = torch.from_numpy(numpy.array(long_feature)).float().to(device)

                # print(f's5 pred shape {out_features.shape}')
                p_classes1 = model1(out_features, long_feature)
                p_classes = p_classes1.squeeze()

                clc_loss = criterion_phase1(p_classes, labels_phase_torch)

                _, preds_phase = torch.max(p_classes.data, 1)
                loss_phase = criterion_phase1(p_classes, labels_phase_torch)

                val_loss_phase += loss_phase.data.item()

                val_corrects_phase += torch.sum(preds_phase == labels_phase_torch.data)
                val_acc_each_video.append(float(torch.sum(preds_phase == labels_phase_torch.data))/val_num_each_80[i])
                # TODO

                for j in range(len(preds_phase)):
                    val_all_preds_phase.append(int(preds_phase.data.cpu()[j]))
                for j in range(len(labels_phase_torch)):
                    val_all_labels_phase.append(int(labels_phase_torch.data.cpu()[j]))

                val_progress += 1
                if val_progress * batch_size >= len(val_we_use_start_idx_80):
                    percent = 100.0
                    print('Val progress: %s [%d/%d]' % (str(percent) + '%', len(val_we_use_start_idx_80),
                                                        len(val_we_use_start_idx_80)), end='\n')
                else:
                    percent = round(val_progress * batch_size / len(val_we_use_start_idx_80) * 100, 2)
                    print('Val progress: %s [%d/%d]' % (str(percent) + '%', val_progress * batch_size, len(val_we_use_start_idx_80)),
                        end='\r')

        #evaluation only for training reference
        val_elapsed_time = time.time() - val_start_time
        val_accuracy_phase = float(val_corrects_phase) / len(val_labels_80)
        val_acc_video = numpy.mean(val_acc_each_video)
        val_average_loss_phase = val_loss_phase

        val_recall_phase = metrics.recall_score(val_all_labels_phase, val_all_preds_phase, average='macro')
        val_precision_phase = metrics.precision_score(val_all_labels_phase, val_all_preds_phase, average='macro')
        val_jaccard_phase = metrics.jaccard_score(val_all_labels_phase, val_all_preds_phase, average='macro')
        val_precision_each_phase = metrics.precision_score(val_all_labels_phase, val_all_preds_phase, average=None)
        val_recall_each_phase = metrics.recall_score(val_all_labels_phase, val_all_preds_phase, average=None)

        test_progress = 0
        test_corrects_phase = 0
        test_all_preds_phase = []
        test_all_labels_phase = []
        test_acc_each_video = []
        test_start_time = time.time()

        with torch.no_grad():
            for i in test_we_use_start_idx_80:
                labels_phase = np.array([])
                labels_phase_torch = []
                for j in range(test_start_vidx[i], test_start_vidx[i]+test_num_each_80[i]):
                    labels_phase = np.append(labels_phase, np.copy(test_labels_80[j][0]))
                    labels_phase_torch.append(test_labels_80[j][0])
                labels_phase_torch = torch.LongTensor(labels_phase_torch).to(device)

                long_feature = get_long_feature(start_index=test_start_vidx[i],
                                                lfb=g_LFB_test, LFB_length=test_num_each_80[i])

                # Prepare data in the foramt of S5
                video_fe = long_feature
                inputs = np.array(video_fe).astype(float)
                labels = labels_phase.astype(float)
                integration_times = None

                # Evaluation step
                loss, acc, pred = eval_step(inputs, labels, integration_times, state, model_cls, args.batchnorm)

                out_features = np.transpose(pred, axes=(1,0))
                out_features = torch.tensor(np.expand_dims(out_features, axis=0).tolist(), device=device).detach()
                long_feature = torch.from_numpy(numpy.array(long_feature)).float().to(device)

                # print(f's5 pred shape {out_features.shape}')

                p_classes1 = model1(out_features, long_feature)
                p_classes = p_classes1.squeeze()

                clc_loss = criterion_phase1(p_classes, labels_phase_torch)

                _, preds_phase = torch.max(p_classes.data, 1)

                test_corrects_phase += torch.sum(preds_phase == labels_phase_torch.data)
                test_acc_each_video.append(float(torch.sum(preds_phase == labels_phase_torch.data)) / test_num_each_80[i])
                # TODO

                for j in range(len(preds_phase)):
                    test_all_preds_phase.append(int(preds_phase.data.cpu()[j]))
                for j in range(len(labels_phase_torch)):
                    test_all_labels_phase.append(int(labels_phase_torch.data.cpu()[j]))

                test_progress += 1
                if test_progress * batch_size >= len(test_we_use_start_idx_80):
                    percent = 100.0
                    print('Test progress: %s [%d/%d]' % (str(percent) + '%', len(test_we_use_start_idx_80),
                                                        len(test_we_use_start_idx_80)), end='\n')
                else:
                    percent = round(test_progress * batch_size / len(test_we_use_start_idx_80) * 100, 2)
                    print('Test progress: %s [%d/%d]' % (
                    str(percent) + '%', test_progress * batch_size, len(test_we_use_start_idx_80)),
                        end='\r')

        test_accuracy_phase = float(test_corrects_phase) / len(test_labels_80)
        test_acc_video = numpy.mean(test_acc_each_video)
        test_elapsed_time = time.time() - test_start_time

        print('epoch: {:4d}'
            ' train in: {:2.0f}m{:2.0f}s'
            ' train loss(phase): {:4.4f}'
            ' train accu(phase): {:.4f}'
            ' valid in: {:2.0f}m{:2.0f}s'
            ' valid loss(phase): {:4.4f}'
            ' valid accu(phase): {:.4f}'
            ' valid accu(video): {:.4f}'
            ' test in: {:2.0f}m{:2.0f}s'
            ' test accu(phase): {:.4f}'
            ' test accu(video): {:.4f}' 
            .format(epoch,
                    train_elapsed_time // 60,
                    train_elapsed_time % 60,
                    train_average_loss_phase,
                    train_accuracy_phase,
                    val_elapsed_time // 60,
                    val_elapsed_time % 60,
                    val_average_loss_phase,
                    val_accuracy_phase,
                    val_acc_video,
                    test_elapsed_time // 60,
                    test_elapsed_time % 60,
                    test_accuracy_phase,
                    test_acc_video))

        print("val_precision_each_phase:", val_precision_each_phase)
        print("val_recall_each_phase:", val_recall_each_phase)
        print("val_precision_phase", val_precision_phase)
        print("val_recall_phase", val_recall_phase)
        print("val_jaccard_phase", val_jaccard_phase)

        if val_accuracy_phase > best_val_accuracy_phase:
            best_val_accuracy_phase = val_accuracy_phase
            correspond_train_acc_phase = train_accuracy_phase
            best_model_wts = copy.deepcopy(model1.state_dict())
            best_epoch = epoch
            best_test_all_preds_phase = test_all_preds_phase

        save_val_phase = int("{:4.0f}".format(best_val_accuracy_phase * 10000))
        save_train_phase = int("{:4.0f}".format(correspond_train_acc_phase * 10000))
        base_name = "S4" \
                    + "_length_" + str(sequence_length) \
                    + "_epoch_" + str(best_epoch) \
                    + "_train_" + str(save_train_phase) \
                    + "_val_" + str(save_val_phase)

    if not os.path.exists("./eval"):
        os.makedirs("./eval")
    pred_name = './eval/s5_trans.pkl'
    with open(pred_name, 'wb') as f:
        pickle.dump(best_test_all_preds_phase, f)
    '''
    pred_score_name = './eval/python/Trans_SV_score_new.pkl'
    with open(pred_score_name, 'wb') as f:
        pickle.dump(test_all_labels_phase, f)
    '''
    model_name = 'S4'
    if not os.path.exists("./best_model"):
        os.makedirs("./best_model")
    if not os.path.exists("./best_model/S4"):
        os.makedirs("./best_model/S4/")
    torch.save(best_model_wts, "./best_model/S4/" + model_name + "_weights" + ".pth")
    print("best_epoch", str(best_epoch))


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
    parser.add_argument('--split_val_test', type=str2bool, default=False, help='Whether to split the test set into 8/32 for validation')
    parser.add_argument('--save_weights', type=str2bool, default=False, help='Whether to save weights')
    parser.add_argument('--restore_path', type=str, default='', help='Path to restore state from')

    train_s5_trans(parser.parse_args())
