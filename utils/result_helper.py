import pandas as pd
import matplotlib.pyplot as plt

def save_train_plots(history):
    plt.title('Loss')
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='val')
    plt.legend()
    # plt.show()
    plt.savefig('results/figures/loss.png')

    plt.figure()
    plt.title('Accuracy')
    plt.plot(history.history['accuracy'], label='train')
    plt.plot(history.history['val_accuracy'], label='val')
    plt.legend()
    # plt.show()
    plt.savefig('results/figures/accuracy.png')

def results_to_submit(results,save_path,id_series,class_label_to_class_dict,rank_label_to_rank_dict,class_label_to_rank_label_dict):
    df=pd.DataFrame(results,columns=['class_label']) # class id
    df['id']=id_series # id
    df['rank_label']=df['class_label'].apply(lambda x: class_label_to_rank_label_dict[x]) # to rank id
    df['class_label']=df['class_label'].apply(lambda x: class_label_to_class_dict[x]) # to class string
    df['rank_label']=df['rank_label'].apply(lambda x: rank_label_to_rank_dict[x]) # to rank string

    df_submit=df[['id','class_label','rank_label']]
    df_submit.to_csv(save_path,index=False)