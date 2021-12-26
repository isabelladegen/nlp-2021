from datasets import load_metric


def final_score_for(predictions: [], references: []):
    metric = load_metric("squad_v2")
    metric.add_batch(predictions=predictions, references=references)
    final_score = metric.compute()
    return final_score
