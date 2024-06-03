from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

def plot_scaling_nlp(save : bool = False):
    sample_data = {
        "llama2-7b": {
            "ROUGE-L": 0.151,
            "BLEU": 0.043,
            "METEOR": 0.168,
            "BERTScore":0.672
        },
        "llama2-13b": {
            "ROUGE-L": 0.351,
            "BLEU": 0.163,
            "METEOR": 0.383,
            "BERTScore": 0.728
        },
        "llama2-70b": {
            "ROUGE-L": 0.343,
            "BLEU": 0.078,
            "METEOR": 0.317,
            "BERTScore": 0.734
        }
    }

    colors = ["red","green","blue","orange"]

    # Extracting metric names and values
    metrics = list(next(iter(sample_data.values())).keys())
    llm_names = list(sample_data.keys())
    values = list(sample_data.values())

    # Plotting
    plt.figure(figsize=(7, 5))
    plt.gca().set_facecolor('gainsboro')
    
    
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    
    plt.grid(color='white') 

    for i in range(len(metrics)):
        metric_values = [llm[metrics[i]] for llm in values]
        print(metric_values)
        plt.plot(llm_names, metric_values, marker='o', label=metrics[i], linewidth=1, color=colors[i])

    plt.title('Change in metric values for Llama-2 when scaling up')
    plt.xlabel('Models')
    plt.ylabel('Metric Values')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    

    if save:
        pth = "img/metrics_change_scaling.png"
        plt.savefig(pth,dpi=1200)
        print(f"image saved at {pth}")
    else:
        plt.show()

def plot_scaling_time(save : bool = False):
    sample_data = {
        "llama2-7b":8.886,
        "llama2-13b":13.999,
        "llama2-70b":34.824
    }


    # Extracting labels and values from sample_data
    labels = list(sample_data.keys())
    values = list(sample_data.values())

    # Creating the line plot
    plt.figure(figsize=(7, 5))

    plt.gca().set_facecolor('gainsboro')
    
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)

    plt.plot(labels, values, marker='o', linestyle='-', linewidth=1,color="black")

    # Adding title and labels
    plt.title('Changes in processing time for Llama2 when scaling up')
    plt.xlabel('Models')
    plt.ylabel('Average time per iteration (s/it)')

    plt.grid(color='white')

    # Rotating x-axis labels for better readability
    plt.xticks(rotation=45)

    # Displaying the plot
    plt.tight_layout()
    
    if save:
        pth = "img/time_change_scaling.png"
        plt.savefig(pth,dpi=1200)
        print(f"image saved at {pth}")
    else:
        plt.show()

def plot_training(path:str, save_path:str = None):
    data = pd.read_csv(path)
    train_loss = data['train/loss']
    train_loss.dropna(inplace=True)
    eval_loss = data['eval/loss']
    eval_loss.dropna(inplace=True)
    
    colors = ["blue","orange"]

    # Plotting
    plt.figure(figsize=(7, 5))
    plt.gca().set_facecolor('gainsboro')
    
    
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    plt.plot(train_loss.index, train_loss.values, marker=None, label='train loss', linewidth=1.5, color=colors[0])
    plt.plot(eval_loss.index, eval_loss.values, marker=None, label='eval loss', linewidth=1.5, color=colors[1])
    
    plt.grid(color='white') 

    
    plt.title('Loss during finetuning')
    plt.xlabel('Global steps')
    plt.ylabel('Loss')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path,dpi=1200)
        print(f"image saved at {save_path}")
    else:
        plt.show()

def plot_score_compare(save_path:str = None):

    x_values = [1.2,5.2,9.2]
    sweditron = [0.435,0.225,0.466]
    vx_values = [2.8,6.8,10.8]
    vanVeen = [0.64,0.433,0]
    tx_values =[2,6,10] 
    tang = [0.215,0.108,0.311]
    colors = ["green","blue","orange"]

    plt.figure(figsize=(7, 5))
    plt.gca().set_facecolor('gainsboro')
    
    plt.grid(color='white',zorder=0) 

    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    plt.bar(x_values,sweditron,label='Our model',color=colors[0],zorder=3)
    plt.bar(tx_values,tang,label='Tang',color=colors[1],zorder=3)
    plt.bar(vx_values,vanVeen,label='van Veen',color=colors[2],zorder=3)

    

    plt.title('Metrics comparison to similar projects')
    #plt.xlabel('Global steps')
    plt.ylabel('Metric score')
    plt.legend()
    x = [2,6,10]
    my_xticks = ["ROUGE-L","BLEU","METEOR"]
    plt.xticks(x,my_xticks,rotation=0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path,dpi=1200)
        print(f"image saved at {save_path}")
    else:
        plt.show()


#plot_scaling_nlp(save=True)
#plot_scaling_time(save=True)
#plot_training('img/fine-tuning.csv',save_path='img/fine-tuning.png')
plot_score_compare('img/project_compare.png')