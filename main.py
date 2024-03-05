from time import time
from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader, get_eval_loader
from wilds.common.grouper import CombinatorialGrouper
import torchvision.transforms as transforms
import torch
from transformers import AutoImageProcessor, ResNetForImageClassification, ResNetConfig
import torch.optim as optim
import argparse
import numpy as np
import json
import torch.nn as nn
from wilds.common.metrics import GroupDRO
from torch.utils.tensorboard import SummaryWriter

class ModifiedResNetForImageClassification(ResNetForImageClassification):
    def __init__(self, config: ResNetConfig):
        super().__init__(config)
        num_features = self.classifier[-1].in_features
        self.classifier[-1] = nn.Linear(num_features, 182)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, **kwargs):
        outputs = super().forward(x, **kwargs)
        logits = outputs.logits
        logits = self.softmax(logits)
        return logits

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
writer = SummaryWriter()

# def IRM_loss(model, inputs, outputs):
#     gradients = torch.autograd.grad(outputs=outputs, inputs=inputs, grad_outputs=torch.ones_like(outputs), create_graph=True)[0]
    
#     penalty = torch.mean(torch.sum(torch.square(gradients), dim=1))
    
#     return penalty

if __name__ == '__main__':
    # Set argument Parsers
    dataset = get_dataset(dataset="iwildcam", download=False)
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--epoch", type=int,default=12)
    parser.add_argument("--batch_size",type=int,default=16)
    parser.add_argument("--group",type=int,default=1)
    parser.add_argument("--subset_size", type=float,default=0.05)

    # hyperparameter sweep [1e-5,1e-4,5e-6]
    args = parser.parse_args()
    config = ResNetConfig.from_pretrained("microsoft/resnet-50")
    model = ModifiedResNetForImageClassification(config)
    model.to(device)
    eval_model = ModifiedResNetForImageClassification(config)
    eval_model.to(device)
    # criterion
    criterion = torch.nn.CrossEntropyLoss()

    # optimizer
    optimizer = optim.Adam(model.parameters(),lr=args.learning_rate,weight_decay=1e-7)
    
    # Get the training set
    train_data = dataset.get_subset(
        "train",
        transform=transforms.Compose(
            [transforms.Resize((448, 448)), transforms.ToTensor()]
        ),
    )
    #print(dataset['from_source_domain'])
    grouper = CombinatorialGrouper(dataset, ['location']) 
    if args.group==1:
        train_loader = get_train_loader("group",train_data,n_groups_per_batch=16,grouper=grouper,batch_size=args.batch_size)
        print('loaded')
    else:
        train_loader = get_train_loader("standard", train_data, batch_size=args.batch_size)
    # test_dataset = get_dataset(dataset="iwildcam", download=False, unlabeled=True)
    # unlabeled_data = test_dataset.get_subset(
    #     "test_unlabeled",
    #     transform=transforms.Compose(
    #         [transforms.Resize((448, 448)), transforms.ToTensor()]
    #     ),
    # )
    # unlabeled_loader = get_train_loader("standard", unlabeled_data, batch_size=args.batch_size)
    print(f"size of training loader {len(train_loader)}")
    ood_val_data = dataset.get_subset("val",transform=transforms.Compose(
            [transforms.Resize((448, 448)), transforms.ToTensor()]
        ))
    # Prepare the data loader
    ood_val_loader = get_eval_loader("standard", ood_val_data,batch_size=args.batch_size)

    id_val_data = dataset.get_subset("id_val",transform=transforms.Compose(
            [transforms.Resize((448, 448)), transforms.ToTensor()]
        ))
    # Prepare the data loader
    id_val_loader = get_eval_loader("standard", id_val_data,batch_size=args.batch_size)
    print(f'size of id validation dataset: {len(id_val_loader)}, and ood: {len(ood_val_loader)}')
    ood_test_data = dataset.get_subset("test",transform=transforms.Compose(
            [transforms.Resize((448, 448)), transforms.ToTensor()]
        ))
    # Prepare the data loader
    ood_test_loader = get_eval_loader("standard", ood_test_data,batch_size=args.batch_size)

    id_test_data = dataset.get_subset("id_test",transform=transforms.Compose(
            [transforms.Resize((448, 448)), transforms.ToTensor()]
        ))
    # Prepare the data loader
    id_test_loader = get_eval_loader("standard", id_test_data,batch_size=args.batch_size)
    print(f'size of id test dataset: {len(id_test_loader)}, and ood: {len(ood_test_loader)}')

    best_id_val_loss = float("inf")
    best_ood_val_loss = float('inf')

    # results saving
    records_id = []
    records_ood = []

    # Train loop
    training_loss_list = []
    id_val_loss_list = []
    ood_val_loss_list = []


    for epoch in range(args.epoch):
        print(f'Epoch: {epoch}')
        training_loss = 0
        model.train()
        start_time = time()
        for batch_idx, labeled_batch in enumerate(train_loader):
            if batch_idx>=len(train_loader)*args.subset_size:
                print(f'one epoch needs time{time()-start_time}')
                break
            x, y, metadata = labeled_batch
            #unlabeled_x, unlabeled_metadata = unlabeled_batch
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            y_prediction = model(x)
            loss = criterion(y_prediction,y)
            
            loss.backward()
            optimizer.step()

            training_loss += loss.item()
            
        epoch_loss = training_loss / (batch_idx+1)
        writer.add_scalar("Loss/train",epoch_loss,epoch)
        print(f'training_loss: {epoch_loss}')

        training_loss_list.append(epoch_loss)
        model.eval()
        id_val_loss = 0
        # ID val and testing
        with torch.no_grad():
            for batch_idx, val_batch in enumerate(id_val_loader):
                x,y,_ = val_batch
                if batch_idx >=len(id_val_loader)*args.subset_size:
                    break
                x = x.to(device)
                y = y.to(device)

                y_prediction = model(x)
                
                loss = criterion(y_prediction,y)
            
                id_val_loss += loss.item()

        id_val_loss /= batch_idx+1
        writer.add_scalar("Loss/id_val", id_val_loss,epoch)
        id_val_loss_list.append(id_val_loss)
        if id_val_loss <= best_id_val_loss:
            best_id_val_loss = id_val_loss
            torch.save(model.state_dict(),'best_id_model_wts/best_id_model.pt')


        # validate on OOD validation dataset
        model.eval()
        ood_val_loss = 0
        with torch.no_grad():
            for batch_idx,val_batch in enumerate(ood_val_loader):
                x,y,meta = val_batch
                if batch_idx>=len(ood_val_loader)*args.subset_size:
                    break
                x = x.to(device)
                y = y.to(device)
                y_prediction = model(x)
                loss = criterion(y_prediction,y)
                ood_val_loss += loss.item()

        ood_val_loss /= batch_idx+1
        writer.add_scalar("Loss/ood_val",ood_val_loss,epoch)
        ood_val_loss_list.append(ood_val_loss)
        if ood_val_loss <= best_ood_val_loss:
            best_ood_val_loss = ood_val_loss
            torch.save(model.state_dict(),'best_ood_model_wts/best_ood_model.pt')

        # Testing ID dataset
        eval_model.eval()
        eval_model.load_state_dict(torch.load("best_id_model_wts/best_id_model.pt"))
        # ID val and testing
        model.eval()
        all_y_pred= []
        all_y_true = []
        all_meta = []
        with torch.no_grad():
            for batch_idx, test_batch in enumerate(id_test_loader):
                x,y,metadata = test_batch
                x = x.to(device)
                y = y.to(device)
                if batch_idx>=len(id_test_loader)*args.subset_size:
                    break

                y_prediction = model(x)
                y_prediction = torch.argmax(y_prediction,dim=1)

                all_y_pred.extend(y_prediction.detach().cpu().numpy().tolist())
                all_y_true.extend(y.detach().cpu().numpy().tolist())
                all_meta.extend(metadata)
        #print(f'y true tensor{all_y_true_tensor.shape},y prediction tensor: {all_y_pred_tensor.shape}')
        results,_ = dataset.eval(torch.tensor(all_y_pred),torch.tensor(all_y_true),all_meta)
        writer.add_scalar("test/id_f1",results['F1-macro_all'],epoch)
        writer.add_scalar("test/id_accuracy",results['acc_avg'],epoch)
        writer.add_scalar("test/id_recall",results['recall-macro_all'],epoch)
        records_id.append(results)

        # eval_model.eval()
        # eval_model.load_state_dict(torch.load("best_ood_model_wts/best_ood_model.pt"))
        # OOD val and testing
        all_y_pred= []
        all_y_true = []
        all_meta = []
        with torch.no_grad():
            for batch_idx, test_batch in enumerate(ood_test_loader):
                x,y,metadata = test_batch
                x = x.to(device)
                y = y.to(device)
                if batch_idx>=len(ood_test_loader)*args.subset_size:
                    break

                y_prediction = model(x)
                y_prediction = torch.argmax(y_prediction,dim=1)
                all_y_pred.extend(y_prediction.detach().cpu().numpy().tolist())
                all_y_true.extend(y.detach().cpu().numpy().tolist())
                all_meta.extend(metadata)
        #print(f'y true tensor{all_y_true_tensor.shape},y prediction tensor: {all_y_pred_tensor.shape}')
        results,_ = dataset.eval(torch.tensor(all_y_pred),torch.tensor(all_y_true),all_meta)
        writer.add_scalar("test/ood_f1",results['F1-macro_all'],epoch)
        writer.add_scalar("test/ood_accuracy",results['acc_avg'],epoch)
        writer.add_scalar("test/ood_recall",results['recall-macro_all'],epoch)
        records_ood.append(results)
    writer.flush()
    writer.close()
    with open(f"uniform_grouping_{args.group}_records.json","w") as file_write:
        record = {'group_uniform_sample':args.group,
        'batch':args.batch_size,
        'learning_rate':args.learning_rate,
        'id_results':records_id,
        'epoch':args.epoch,
        'ood_results':records_ood,
        'training_loss':training_loss_list,
        'id_val_loss':id_val_loss_list,
        'ood_val_loss':ood_val_loss_list
        }
        json.dump(record,file_write)







# for x, y_true, metadata in test_loader:
#     y_pred = model(x)
#     #[ accumulate y_true, y_pred, metadata]
# # Evaluate
# dataset.eval(all_y_pred, all_y_true, all_metadata)