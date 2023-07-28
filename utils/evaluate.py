import torch

import sacrebleu
from sacrebleu.metrics import BLEU
import bert_score
from nltk.translate.bleu_score import sentence_bleu
from torchmetrics import BLEUScore, SacreBLEUScore
from torchmetrics.text.bert import BERTScore
import rouge as rouge_lib

# from utils.tokenizer import respond_only

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# def evaluate(model, data, tokenizer, device, max_length):
#     test_questions = df_test['questions'].values
#     test_answers = df_test['answers'].values

#     preds = []
#     for x in test_questions:
#         preds.append(respond_only(model, str(x), tokenizer, tokenizer, device, max_length=max_length))
    
#     sacrebleu_score = sacrebleu.corpus_bleu(preds, test_questions.tolist())
#     bleu_scores = calculate_bleu(preds, test_questions, test_answers)

def calculate_rouge(preds, real):
    scorer = rouge_lib.Rouge()
    scorer.get_scores(preds,
                    real.tolist(),
                    avg=True,
                    ignore_empty=True)
    return scorer
    
def calculate_bertscore(preds, real):
    bertscorer = bert_score.BERTScorer(idf=True,
                      lang='en',
                      rescale_with_baseline=True,
                      use_fast_tokenizer=True,
                      device='cuda')

    bertscorer.compute_idf([r for rs in preds for r in rs])
    prf = bertscorer.score(preds, real.tolist(), batch_size=64)
    return {key: scores.mean().item() for key, scores in zip(('p', 'r', 'f'), prf)}
    
def calculate_bleu(preds, questions, answers):
    bleu_score_1 = 0
    bleu_score_2 = 0
    bleu_score_3 = 0
    bleu_score_4 = 0
    bleu_score_all = 0

    num_of_rows_calculated = 0

    for i, (question, real_answer) in enumerate(zip(questions, answers)):
        try:
            refs = [real_answer.split(' ')]
            hyp = preds[i].split(' ')

            bleu_score_1 += sentence_bleu(refs, hyp, weights=(1,0,0,0))
            bleu_score_2 += sentence_bleu(refs, hyp, weights=(0,1,0,0))
            bleu_score_3 += sentence_bleu(refs, hyp, weights=(0,0,1,0))
            bleu_score_4 += sentence_bleu(refs, hyp, weights=(0,0,0,1))
            bleu_score_all += sentence_bleu(refs, hyp, weights=(.25,.25,.25,.25))

            num_of_rows_calculated+=1
        except:
            continue

    results = {"1-gram": (bleu_score_1/num_of_rows_calculated),
                "2-gram": (bleu_score_2/num_of_rows_calculated),
                "3-gram": (bleu_score_3/num_of_rows_calculated),
                "4-gram": (bleu_score_all/num_of_rows_calculated)}
    
    return results