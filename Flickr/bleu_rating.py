from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

reference = []

reference.append('a girl running in the grass'.split(' '))
reference.append('a girl in a dress is walking'.split(' '))
candidate = 'a girl in a dress is jumping in the grass'.split(' ')

smoothie = SmoothingFunction().method4
score = sentence_bleu(reference, candidate,smoothing_function=smoothie)
print(score)
