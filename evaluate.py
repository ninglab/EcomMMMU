import fire
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score
from utils.utils import *
from utils.template import *

import warnings
warnings.filterwarnings("ignore")

def cpt_bicls(label_list, prediction_list):
    skipped = 0
    
    filtered_prediction_list = []
    filtered_label_list = []
    valid_response = set([i.lower() for i in label_list])
    for i in range(len(prediction_list)):
        pred = prediction_list[i].lower()
        if pred in valid_response:
            filtered_prediction_list.append(pred)
            filtered_label_list.append(label_list[i].lower())
        else:
            skipped += 1
            
    if len(filtered_prediction_list) > 0:
        acc = accuracy_score(filtered_label_list, filtered_prediction_list)
        acc = acc*len(filtered_label_list)/len(prediction_list)
    else:
        acc = 0
    pre = precision_score(filtered_label_list, filtered_prediction_list, pos_label = 'yes')
    rec = recall_score(filtered_label_list, filtered_prediction_list, pos_label = 'yes')
    f1 = f1_score(filtered_label_list, filtered_prediction_list, pos_label = 'yes')

    print(f'& {acc:.3f} & {pre:.3f} & {rec:.3f} & {f1:.3f} & {skipped}')

def cpt_mcls(label_list, prediction_list):
    skipped = 0
    filtered_prediction_list = []
    filtered_label_list = []
    valid_response = set(label_list)
    for i in range(len(prediction_list)):
        pred = prediction_list[i]
        if pred in valid_response:
            filtered_prediction_list.append(pred)
            filtered_label_list.append(label_list[i])
        else:
            skipped += 1
            
    if len(filtered_prediction_list) > 0:
        acc = accuracy_score(filtered_label_list, filtered_prediction_list)
        acc = acc*len(filtered_label_list)/len(prediction_list)
    else:
        acc = 0
    pre_macro = precision_score(filtered_label_list, filtered_prediction_list, average = 'macro')
    rec_macro = recall_score(filtered_label_list, filtered_prediction_list, average = 'macro')
    f1_macro  = f1_score(filtered_label_list, filtered_prediction_list, average = 'macro')

    print(f'& {acc:.3f} & {pre_macro:.3f} & {rec_macro:.3f} & {f1_macro:.3f} & {skipped}')

def cpt_rec1(label_list, prediction_list):
    skipped = 0
    filtered_prediction_list = []
    filtered_label_list = []
    valid_response = set(label_list)

    for i in range(len(prediction_list)):
        pred = prediction_list[i]
        if pred in valid_response:
            filtered_prediction_list.append(pred)
            filtered_label_list.append(label_list[i])
        else:
            skipped += 1
    
    if len(filtered_prediction_list) > 0:
        rec1 = accuracy_score(filtered_label_list, filtered_prediction_list)
        rec1 = rec1*len(filtered_label_list)/len(prediction_list)
    else:
        rec1 = 0
    
    print(f'& {rec1:.3f} & {skipped}')

def evaluate_task(task):
    prediction_list = load_json(f'results/{task}.json')
    data = pd.DataFrame(load_json('data/downstrem_test.json'))
    data = data[data['task'] == task]
    label_list = [row['output'] for _, row in data.iterrows()]

    if task in ['answerability_prediction', 'production_substitute_identification', 'click-through_predction']:
        cpt_bicls(label_list, prediction_list)
    elif task in ['sequential_recommendation']:
        cpt_rec1(label_list, prediction_list)
    elif task in ['multiclass_product_classification', 'binary-question_answering', 'product_relation_prediction', 'sentiment_analysis']:
        cpt_mcls(label_list, prediction_list)

if __name__ == '__main__':
    fire.Fire(evaluate_task)