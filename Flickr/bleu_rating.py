from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

reference = []

reference.append('a person is near a car'.split(' '))
#reference.append('a girl in a dress is walking'.split(' '))
candidate = 'a person is near a car'.split(' ')

smoothie = SmoothingFunction().method4
score = sentence_bleu(reference, candidate,smoothing_function=smoothie)
print(score)
