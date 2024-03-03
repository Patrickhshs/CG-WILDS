from time import time
from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader, get_eval_loader
import torchvision.transforms as transforms
import torch
from transformers import AutoImageProcessor, ResNetForImageClassification
import torch.optim as optim
import argparse
import numpy as np
import json

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

def IRM_loss(model, inputs, outputs):
    # Compute gradients of outputs with respect to inputs
    gradients = torch.autograd.grad(outputs=outputs, inputs=inputs, grad_outputs=torch.ones_like(outputs), create_graph=True)[0]
    
    # Compute the penalty term as the squared norm of gradients
    penalty = torch.mean(torch.sum(torch.square(gradients), dim=1))
    
    return penalty

if __name__ == '__main__':
    # Set argument Parsers
    dataset = get_dataset(dataset="iwildcam", download=False)
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--epoch", type=int,default=3)
    parser.add_argument("--batch_size",type=int,default=32)
    parser.add_argument("--group",type=bool,default=False)
    parser.add_argument("--subset_size", type=float,default=0.3)

    # hyperparameter sweep [1e-5,1e-4,5e-6]
    args = parser.parse_args()
    model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")
    model.to(device)
    # criterion
    criterion = torch.nn.CrossEntropyLoss()

    # optimizer
    optimizer = optim.Adam(model.parameters(),lr=args.learning_rate,weight_decay=1e-3)
    
    # Get the training set
    train_data = dataset.get_subset(
        "train",
        transform=transforms.Compose(
            [transforms.Resize((448, 448)), transforms.ToTensor()]
        ),
    )
    if args.group:
        train_loader = get_train_loader("group",train_data,batch_size=args.batch_size)
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
        model.train(True)
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
            y_prediction = model(x)["logits"]
            y_prediction = torch.softmax(y_prediction,dim=1)
            loss = criterion(y_prediction,y)

            loss.backward()
            optimizer.step()

            training_loss += loss.item()
            
        epoch_loss = training_loss / len(train_loader)
        print(f'training_loss: {epoch_loss}')

        training_loss_list.append(epoch_loss)
        model.eval()
        id_val_loss = 0
        # ID val and testing
        with torch.no_grad():
            for batch_idx, val_batch in enumerate(id_val_loader):
                x,y,_ = val_batch
                # if batch_idx >=100:
                #     break
                x = x.to(device)
                y = y.to(device)

                y_prediction = model(x)['logits']
                y_prediction = torch.softmax(y_prediction,dim=1)
                loss = criterion(y_prediction,y)
                id_val_loss += loss.item()

        id_val_loss /= len(id_val_loader)
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
                # if batch_idx>=1:
                #     break
                x = x.to(device)
                y = y.to(device)
                y_prediction = model(x)['logits']
                y_prediction = torch.softmax(y_prediction,dim=1)
                loss = criterion(y_prediction,y)
                id_val_loss += loss.item()

        ood_val_loss /= len(ood_val_loader)
        ood_val_loss_list.append(ood_val_loss)
        if ood_val_loss <= best_ood_val_loss:
            best_ood_val_loss = ood_val_loss
            torch.save(model.state_dict(),'best_ood_model_wts/best_ood_model.pt')

        # Testing ID dataset
        model.eval()
        model.load_state_dict(torch.load("best_id_model_wts/best_id_model.pt"))
        id_test_loss = 0
        # ID val and testing
        all_y_pred= []
        all_y_true = []
        all_meta = []
        with torch.no_grad():
            for batch_idx, test_batch in enumerate(id_test_loader):
                x,y,metadata = test_batch
                x = x.to(device)
                y = y.to(device)
                if batch_idx>=1:
                    break

                y_prediction = model(x)['logits']
                y_prediction = torch.argmax(torch.softmax(y_prediction,dim=1),dim=1)

                all_y_pred.append(y_prediction.detach().cpu().numpy())
                all_y_true.append(y.detach().cpu().numpy())
                all_meta.append(metadata)
        all_y_pred_tensor = torch.squeeze(torch.tensor(np.array(all_y_pred)))
        all_y_true_tensor = torch.squeeze(torch.tensor(np.array(all_y_true)))
        print(f'y true tensor{all_y_true_tensor.shape},y prediction tensor: {all_y_pred_tensor.shape}')
        results,_ = dataset.eval(all_y_pred_tensor,all_y_true_tensor,all_meta)
        records_id.append(results)

        model.eval()
        model.load_state_dict(torch.load("best_ood_model_wts/best_ood_model.pt"))
        # ID val and testing
        all_y_pred= []
        all_y_true = []
        all_meta = []
        with torch.no_grad():
            for batch_idx, test_batch in enumerate(ood_test_loader):
                x,y,metadata = test_batch
                x = x.to(device)
                y = y.to(device)
                if batch_idx>=1:
                    break

                y_prediction = model(x)['logits']
                y_prediction = torch.argmax(torch.softmax(y_prediction,dim=1),dim=1)
                all_y_pred.append(y_prediction.detach().cpu().numpy())
                all_y_true.append(y.detach().cpu().numpy())
                all_meta.append(metadata)
        all_y_pred_tensor = torch.squeeze(torch.tensor(np.array(all_y_pred)))
        all_y_true_tensor = torch.squeeze(torch.tensor(np.array(all_y_true)))
        print(f'y true tensor{all_y_true_tensor.shape},y prediction tensor: {all_y_pred_tensor.shape}')
        results,_ = dataset.eval(all_y_pred_tensor,all_y_true_tensor,all_meta)
        records_ood.append(results)
    
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