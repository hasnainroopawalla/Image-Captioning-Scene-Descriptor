from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

token_dir = "Flickr8k_text/Flickr8k.token.txt"

image_captions = open(token_dir).read().split('\n')
caption = {}    
for i in range(len(image_captions)-1):
    id_capt = image_captions[i].split("\t")
    id_capt[0] = id_capt[0][:len(id_capt[0])-2]
    if id_capt[0] in caption:
        caption[id_capt[0]].append(id_capt[1])
    else:
        caption[id_capt[0]] = [id_capt[1]]


def get_bleu(img_path, pred_caption):
    return round(sentence_bleu(caption[img_path], pred_caption),4)
