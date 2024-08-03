import argparse
import numpy as np
import torch
import torch.nn as nn
import random
import time
from sklearn import metrics
import copy


from get_loaders import get_dataloader


###########################
# Model Training Function
###########################
def train_model(
    train_dataloader,
    val_dataloader,
    test_dataloader, 
    model,
    out_features=7,
    num_workers=3,
    batch_size=1,
    learning_rate=1e-3,
    max_epochs=35,
    sequence_length=30,
    device='cpu',
    pretrain_pth = None
):
    # Crossentropyloss is weighted by class probabilities to help with class imbalance  
    weights_train = np.asarray([1.6411019141231247,
                0.19090963801041133,
                1.0,
                0.2502662616859295,
                1.9176363911137977,
                0.9840248158200853,
                2.174635818337618,])
    device = torch.device(device)
    criterion_phase = nn.CrossEntropyLoss(weight=torch.from_numpy(weights_train).float().to(device))

    # Load in pretrained weights
    if not model:
        raise Exception('No model specified.')
    if pretrain_pth:
        model.load_state_dict(torch.load(pretrain_pth))
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # First 40 Cholec80 videos used for training, next 40 are split 8/32 for validation and testing
    best_val_accuracy_phase = 0.0
    best_train_accuracy_phase = 0.0
    best_epoch = 0
    train_vid_num = [x for x in range(40)]
    val_vid_num = [x for x in range(8)]
    test_vid_num = [x for x in range(32)]
    for epoch in range(max_epochs):
        torch.cuda.empty_cache()
        # random.shuffle(train_vid_num)
        model.train()
        train_loss_phase = 0.0
        train_corrects_phase = 0
        batch_progress = 0.0
        running_loss_phase = 0.0
        minibatch_correct_phase = 0.0
        sequence_length = 16            # Model sequence length
        train_start_time = time.time()
        
        for i, data in enumerate(train_dataloader):
            optimizer.zero_grad()
            print(f'Device: {device}')
            inputs, labels = data[0].to(device), data[1].to(device)

            inputs = inputs.view(-1, sequence_length, 3, 224, 224)
            inputs = inputs.transpose(1,2)
            print(f'INPUTS: {inputs.shape}')
            # outputs = model.forward(inputs).data.cpu().numpy()
            outputs = model.forward(inputs)
            print(f'OUTPUTS: {outputs.shape}')
            # p_classes = outputs.squeeze(1).detach()
            # p_classes = outputs.detach()
            p_classes = outputs
            print(f'Devices of p_classes and labels: {p_classes.device}, {labels.device}')
            print(f'Shapes of p_classes and labels: {p_classes.shape}, {labels.shape}')

            clc_loss = criterion_phase(p_classes, labels.to(torch.int64))
            _, preds_phase = torch.max(p_classes.data, 1)
            loss = clc_loss
            loss.backward()
            optimizer.step()

            running_loss_phase += clc_loss.data.item()
            train_loss_phase += clc_loss.data.item()
            batch_corrects_phase = torch.sum(preds_phase == labels.data)
            train_corrects_phase += batch_corrects_phase
            minibatch_correct_phase += batch_corrects_phase

        train_elapsed_time = time.time() - train_start_time
        train_accuracy_phase = float(train_corrects_phase) / len(train_dataloader) # TODO: CHECK
        train_average_loss_phase = train_loss_phase


        raise Exception('train checkpoint')


        # Sets the module in evaluation mode.
        model.eval()
        val_loss_phase = 0.0
        val_corrects_phase = 0
        val_start_time = time.time()
        val_progress = 0
        val_all_preds_phase = []
        val_all_labels_phase = []
        val_acc_each_video = []
        with torch.no_grad():
            for data in val_dataloader:
                if device == 'cuda':
                    inputs, labels = data[0].to(device), data[1].to(device)
                else:
                    inputs,
                inputs = inputs.view(-1, sequence_length, 3, 224, 224)
                inputs = inputs.transpose(1,2)
                # outputs = model.forward(inputs).data.cpu().numpy()
                outputs = model.forward(inputs)[-1]
                p_classes = outputs.squeeze(1).detach()

                clc_loss = criterion_phase(p_classes, labels)
                _, preds_phase = torch.max(p_classes.data, 1)
                loss = clc_loss

                val_loss_phase += clc_loss.data.item()
                val_corrects_phase += torch.sum(preds_phase == labels.data)
                val_acc_each_video.append(float(torch.sum(preds_phase == labels.data))/val_num_each_80[i])

                for j in range(len(preds_phase)):
                    val_all_preds_phase.append(int(preds_phase.data.cpu()[j]))
                for j in range(len(labels)):
                    val_all_labels_phase.append(int(labels.data.cpu()[j]))

        #evaluation only for training reference
        val_elapsed_time = time.time() - val_start_time
        val_accuracy_phase = float(val_corrects_phase) / len(val_all_labels_phase)
        val_acc_video = np.mean(val_acc_each_video)
        val_average_loss_phase = val_loss_phase

        # Calculate metrics
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
            for data in val_dataloader:
                if device == 'cuda':
                    inputs, labels = data[0].to(device), data[1].to(device)
                else:
                    inputs,
                inputs = inputs.view(-1, sequence_length, 3, 224, 224)
                # outputs = model.forward(inputs).data.cpu().numpy()
                outputs = model.forward(inputs)[-1]
                p_classes = outputs.squeeze(1).detach()

                clc_loss = criterion_phase(p_classes, labels)
                _, preds_phase = torch.max(p_classes.data, 1)
                loss = clc_loss

                test_corrects_phase += torch.sum(preds_phase == labels.data)
                test_acc_each_video.append(float(torch.sum(preds_phase == labels.data)) / test_num_each_80[i])
                for j in range(len(preds_phase)):
                    test_all_preds_phase.append(int(preds_phase.data.cpu()[j]))
                for j in range(len(labels)):
                    test_all_labels_phase.append(int(labels.data.cpu()[j]))


        test_accuracy_phase = float(test_corrects_phase) / len(test_labels_80)
        test_acc_video = np.mean(test_acc_each_video)
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
            best_model_wts = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            best_test_all_preds_phase = test_all_preds_phase

        save_val_phase = int("{:4.0f}".format(best_val_accuracy_phase * 10000))
        save_train_phase = int("{:4.0f}".format(correspond_train_acc_phase * 10000))
        base_name = "TeCNO50_trans1_3_5_1" \
                    + "_length_" + str(sequence_length) \
                    + "_epoch_" + str(best_epoch) \
                    + "_train_" + str(save_train_phase) \
                + "_val_" + str(save_val_phase)

'''
pred_name = './eval/python/Trans_SV_new_weights.pkl'
with open(pred_name, 'wb') as f:
    pickle.dump(best_test_all_preds_phase, f)
torch.save(best_model_wts, "./best_model/TeCNO/" + base_name + "_new" + "_weights" + ".pth")
print("best_epoch", str(best_epoch))
'''


def main(args):
    # Set world    
    seed = 1
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Retrieve dataloaders
    train_dataloader, val_dataloader, test_dataloader = get_dataloader(
        data_path = args.data_path,
        label_path = args.label_path,
        seq_length=args.seq_length,
        train_batch_size=args.batch_size,
        val_batch_size=args.batch_size,
        use_flip=args.use_flip,
        crop_type=args.crop_type,
        num_workers=args.workers
    )
    print(f'Length of train_dataloader: {len(train_dataloader)}')
    print(f'Length of val_dataloader: {len(val_dataloader)}')
    print(f'Length of test_dataloader: {len(test_dataloader)}')

    # Model
    from dummy_model import build_cfg_swin, SwinTransformer3D
    cfg = build_cfg_swin(size='base')
    model = SwinTransformer3D(
        embed_dim=cfg['embed_dim'],
        depths=cfg['depths'],
        num_heads=cfg['num_heads'],
        patch_size=cfg['patch_size'],
        window_size=cfg['window_size'],
        drop_path_rate=cfg['drop_path_rate'],
        patch_norm=cfg['patch_norm']
    )
    
    train_model( 
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=test_dataloader, 
        model=model,
        out_features=7,
        num_workers=args.workers,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_epochs=args.epochs,
        sequence_length=args.seq_length,
        device='cpu' if not torch.cuda.is_available() else 'cuda:0',
        pretrain_pth = None
    )
    


if __name__=='__main__':
    # '/home/ppak/surgical_adventure/src/Trans-SVNet/train_val_paths_labels1.pkl'
    parser = argparse.ArgumentParser(description='train cholec80 model')
    parser.add_argument('-d','--data',type=str,required=True,help='Path to train_val pkl file')
    parser.add_argument('-l','--label_path',type=str,required=True,help='Path to train_val pkl file')
    parser.add_argument('-s','--seq_length',type=int,default=1,help='Sequence length')
    parser.add_argument('-b','--batch_size',type=int,default=400,help='Batch size')
    parser.add_argument('-f','--use_flip',type=int,default=1,help='Use flip')
    parser.add_argument('-c','--crop_type',type=int,default=1,help='Use crop')
    parser.add_argument('-w','--workers',type=int,default=4,help='Number of workers')
    parser.add_argument('-e','--epochs',type=int,default=35,help='Number of epochs')
    parser.add_argument('--lr',type=float,default=1e-3,help='Learning rate')
    args = parser.parse_args()
    main(args)