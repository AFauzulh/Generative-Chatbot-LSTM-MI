import re

def normalize(txt):
    txt = txt.lower()
    txt = re.sub(r"i'm", "i am", txt)
    txt = re.sub(r"he's", "he is", txt)
    txt = re.sub(r"she's", "she is", txt)
    txt = re.sub(r"that's", "that is", txt)
    txt = re.sub(r"what's", "what is", txt)
    txt = re.sub(r"where's", "where is", txt)
    txt = re.sub(r"\'ll", " will", txt)
    txt = re.sub(r"\'ve", " have", txt)
    txt = re.sub(r"\'re", " are", txt)
    txt = re.sub(r"\'d", " would", txt)
    txt = re.sub(r"won't", "will not", txt)
    txt = re.sub(r"can't", "can not", txt)
    txt = re.sub(r"a'ight", "alright", txt)
    txt = re.sub(r"n't", ' not', txt)
    return txt

def remove_non_letter(data):
    return re.sub(r'[^a-zA-Z]',' ', data)

def remove_whitespace(data):
    data = [x for x in data.split(' ') if x]
    return ' '.join(data)

def tokenize(text):
    text = str(text)
    return [token for token in text.split(' ')]

def add_sos_eos(text):
    return '<sos> ' + text + ' <eos>'

def preprocess_1(data):
    data = str(data)
    data = normalize(data)
    data = remove_non_letter(data)
    data = remove_whitespace(data)
    return data

def preprocess_2(text):
    text = add_sos_eos(text)
    text = tokenize(text)
    return text

# def preprocess_1(data):
#   data = data.apply(str)
#   data = data.apply(normalize)
#   data = data.apply(remove_non_letter)
#   data = data.apply(remove_whitespace)
#   return data

# def preprocess_1(data):
#     data = data.map(lambda x: normalize(x))
#     data = data.map(lambda x: remove_non_letter(x))
#     data = data.map(lambda x: remove_whitespace(x))
#     return data

# def preprocess_2(text, tokenizer, max_len):
#   text = add_sos_eos(text)
#   text = tokenize(text)
# #   text = tokenizer.text_to_sequence(text)
# #   text = pad_sequences(text, max_len=max_len)
#   return text

# def preprocess_2(text):
#   text = add_sos_eos(text)
#   text = tokenize(text)
#   text = tokenizer.text_to_sequence(text)
#   text = pad_sequences(text, max_len=max_len)
#   return text