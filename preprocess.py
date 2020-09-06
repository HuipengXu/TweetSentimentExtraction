import pandas as pd
import numpy as np
import json
from tqdm import tqdm


def find_all(input_str, search_str):
    l1 = []
    length = len(input_str)
    index = 0
    while index < length:
        i = input_str.find(search_str, index)
        if i == -1:
            return l1
        l1.append(i)
        index = i + 1
    return l1


# Convert training data
def training_to_squad(train_data):
    version = 'v1.0'
    output = []

    for line in tqdm(train_data):
        paragraphs = []

        context = line[1]

        qas = []
        question = line[-1]
        qid = line[0]
        answers = []
        answer = line[2]
        if type(answer) != str or type(context) != str or type(question) != str:
            print(context, type(context))
            print(answer, type(answer))
            print(question, type(question))
            continue
        answer_starts = find_all(context, answer)
        for answer_start in answer_starts:
            answers.append({'answer_start': answer_start, 'text': answer})
        qas.append({'question': question, 'id': qid, 'is_impossible': False, 'answers': answers})

        paragraphs.append({'context': context, 'qas': qas})
        output.append({'title': 'None', 'paragraphs': paragraphs})

    idx = list(range(len(output)))
    np.random.shuffle(idx)
    valid_output = [output[i] for i in idx[:5000]]
    train_output = [output[i] for i in idx[5000:]]

    debug_train = train_output[:1000]
    debug_valid = valid_output[:1000]

    train_output = {'version': version, 'data': train_output}
    valid_output = {'version': version, 'data': valid_output}

    debug_train_output = {'version': version, 'data': debug_train}
    debug_valid_output = {'version': version, 'data': debug_valid}

    with open('data/train.json', 'w') as outfile:
        json.dump(train_output, outfile)

    with open('data/debug_train.json', 'w') as outfile:
        json.dump(debug_train_output, outfile)

    with open('data/valid.json', 'w') as outfile:
        json.dump(valid_output, outfile)

    with open('data/debug_valid.json', 'w') as outfile:
        json.dump(debug_valid_output, outfile)


# Convert test data
def test_to_squad(test_data):
    output = {}
    output['version'] = 'v1.0'
    output['data'] = []

    for line in tqdm(test_data):
        paragraphs = []

        context = line[1]

        qas = []
        question = line[-1]
        qid = line[0]
        if type(context) != str or type(question) != str:
            print(context, type(context))
            print(question, type(question))
            continue
        answers = []
        answers.append({'answer_start': 1000000, 'text': '__None__'})
        qas.append({'question': question, 'id': qid, 'is_impossible': False, 'answers': answers})

        paragraphs.append({'context': context, 'qas': qas})
        output['data'].append({'title': 'None', 'paragraphs': paragraphs})

    with open('data/test.json', 'w') as outfile:
        json.dump(output, outfile)


def generate_debug_data(train_df, valid_df):
    train_df.iloc[:1000].to_csv('./data/debug_train.csv', index=False, encoding='utf8')
    valid_df.iloc[:1000].to_csv('./data/debug_valid.csv', index=False, encoding='utf8')


if __name__ == '__main__':
    data_dir = './data/'

    # pd_train = pd.read_csv(data_dir + 'train.csv')
    # pd_test = pd.read_csv(data_dir + 'test.csv')
    #
    # train = np.array(pd_train)
    # test = np.array(pd_test)
    #
    # training_to_squad(train)
    # test_to_squad(test)

    train_df = pd.read_csv(data_dir + 'train.csv')
    train_df = train_df.sample(frac=1).reset_index(drop=True).dropna()

    train_df.iloc[:5000].to_csv('./data/clean_valid.csv', index=False, encoding='utf8')
    train_df.iloc[5000:].to_csv('./data/clean_train.csv', index=False, encoding='utf8')

    train_df = pd.read_csv(data_dir + 'clean_train.csv')
    valid_df = pd.read_csv(data_dir + 'clean_valid.csv')

    generate_debug_data(train_df, valid_df)
