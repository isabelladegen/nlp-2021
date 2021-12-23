# predict what's the relevant span
# 1 span, top most, all above a certain percentage, with history without history
# input trained model & rc dataset with user questions and gold answers
# output predictions, actual ansers, f1 scores
from train import BatchTrainer



predictions = []
references = []
for example in rc_dataset:
    question_ = example["question"]
    doc_id = example['title']
    # it does better if the user and agent string are left
    # question_ = question_.replace('user:', '')
    # question_ = question_.replace('agent:', '')

    #preprocess question in the same way
    test_doc = simple_preprocess(question_, deacc=True)
    #calculate vector using model for that document
    model = models_for_doc[doc_id]
    vector = model.infer_vector(test_doc)
    #find the most similar document (spans)
    sims = model.dv.most_similar([vector], topn=1)

    spans_dic = raw_training_docs_per_doc[doc_id]
    tag_for_most_likely_answer = sims[0][0]
    most_likely_answer = spans_dic[tag_for_most_likely_answer]

    # print(f'Question: {question_}\n')
    # print(f'Predicted Answer tag {tag_for_answer}: {most_likely_answer}\n')
    # print(f'Correct answer: {example["answers"]}\n')

    id_ = example["id"]
    predictions.append(
        {'id': id_,
         'prediction_text':
             most_likely_answer,
         'no_answer_probability': 0.0
         }
    )

    #just using their answers
    references.append(
        {
            "id": id_,
            "answers": example["answers"],
        }
    )

predictions[:5]