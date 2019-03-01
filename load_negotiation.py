from pathlib import Path
import pandas as pd
import torch

def load_negotiation(path):
    """
    load dataset and make csv files.
    """
    def get_tag(tokens, tag):
        return tokens[tokens.index('<'+ tag + '>') + 1: tokens.index('</' + tag + '>')]

    def preprocessing_dataset(scenario):
        # calc each issue weight:-> to Torch Tensor
        # edit dialogue tags

        value = [int(val) for val in scenario[0][1::2]]
        value = torch.FloatTensor(value)
        value = value / torch.sum(value)
        value = value.tolist()

        dialogue = ' '.join([token for token in scenario[1] if token != '<eos>'])

        return [dialogue, value[0], value[1], value[2]]

    dataset = []

    text = path.read_text('utf-8').split('\n')
    for line in text:
        tokens = line.strip().split() # split into elements
        scenario = []
        # senario should be like: [[input], [partner_input], [dialogue], [output]]

        # for empty list
        if tokens == []:
            continue
        
        for tag in ['input', 'dialogue', 'output']: 
            # Memo: don't care about a partner this time
            scenario.append(get_tag(tokens, tag))
        
        # discard unreached an agreement dialogue
        if ('<disagree>' in scenario[-1]) or ('<disconnect>' in scenario[-1]) or ('<no_agreement>' in scenario[-1]):
            continue
        scenario = preprocessing_dataset(scenario)
        dataset.append(scenario)

    print(f'{path.name}: {len(dataset)} scenarios.')
    df = pd.DataFrame(dataset, columns=['text', 'value1', 'value2', 'value3'])
    df.to_csv(path.name.split('.')[0] + '.csv', index=False)

if __name__ == '__main__':
    r_path = Path('data/negotiate')
    for child_name in ['train.txt', 'test.txt', 'val.txt']:
        child_path = r_path / child_name
        load_negotiation(child_path)
