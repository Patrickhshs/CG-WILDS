import matplotlib.pyplot as plt
import json
import os
import numpy as np
import pandas as pd

def load_data(path):
    with open(path,"r") as fp:
        resource = json.load(fp)
    id_val_loss_list = resource['id_val_loss']
    ood_val_loss_list = resource['ood_val_loss']
    training_loss_list = resource['training_loss']
    id_list = resource['id_results']
    ood_list = resource['ood_results']
    id_accuracy_list  = [i['acc_avg'] for i in id_list]
    id_recall_list  = [i['recall-macro_all'] for i in id_list]
    id_f1_score_list  = [i['F1-macro_all'] for i in id_list]
    ood_accuracy_list  = [i['acc_avg'] for i in ood_list]
    ood_recall_list  = [i['recall-macro_all'] for i in ood_list]
    ood_f1_score_list  = [i['F1-macro_all'] for i in ood_list]
    #data = [id_val_loss_list,ood_val_loss_list,training_loss_list,id_accuracy_list,id_recall_list,id_f1_score_list,ood_accuracy_list,ood_recall_list,ood_f1_score_list]
    return pd.DataFrame({'id_val_loss':id_val_loss_list,
        'ood_val_loss':ood_val_loss_list,
        'training_loss':training_loss_list,
        'id_accuracy':id_accuracy_list,
        'id_recall':id_recall_list,
        'id_f1':id_f1_score_list,
        'ood_accuracy':ood_accuracy_list,
        'ood_recall':ood_recall_list,
        'ood_f1':ood_f1_score_list})


def plot_visualization(data,columns_to_plot,title):
    plt.figure(figsize=(10,6))
    for column in columns_to_plot:
        plt.plot(data.index,data[column],label=column)
    
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.savefig(f"/scratch/js12556/CG-WILDS/visualization/{title}.pdf")


if __name__ == '__main__':
    data = load_data('/scratch/js12556/CG-WILDS/records.json')
    plot_visualization(data,['id_val_loss','ood_val_loss'],'id val loss vs ood val loss')