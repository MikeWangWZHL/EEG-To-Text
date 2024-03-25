from torchmetrics.text import BLEUScore
from torchmetrics.text import WordErrorRate
from torchmetrics.functional.text.rouge import rouge_score


def compute_metrics(preds, labels):
    result={}
    wer = WordErrorRate()
    result['wer']=wer(preds,labels).item()
    rouge_result=compute_rouge(preds,labels)
    for k,v in rouge_result.items():
        result[k]=v.item()
    labels=[[label] for i,label in enumerate(labels)]
    for i in range(1,5):
        bleu=BLEUScore(n_gram=i)
        result[f'bleu-{i}']=bleu(preds,labels).item()
    return result


def compute_rouge(preds, labels):

    metrics={}
    for decoded_label, decoded_pred in zip(labels, preds):
        metric=rouge_score(decoded_pred,decoded_label)
        for key in metric.keys():
            metrics[key]=metrics.get(key, 0) + metric[key]
    for key in metrics.keys():
        metrics[key]=metrics[key]/len(labels)*100

    return metrics