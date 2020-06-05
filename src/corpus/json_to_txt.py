import json
import os

root = 'cord19_data'
dirname = 'cord19'
name = 'cord19'

def convert_json_to_txt(json_path='corpus.json'):
    print('Converting json to text files.')
    print('\nNOTE: corpus.json must exist. Make sure to unzip corpus.json.zip\n')

    # create proper folder (Cord19 Dataset obj reads from this)
    folder = os.path.join(root, os.path.join(dirname, name))
    if not os.path.exists(folder):
        os.makedirs(folder)

    with open(json_path, 'r') as in_file:
        read = in_file.read()
        data = json.loads(read)

        for key, val in data.items():
            if 'tokens' in key:
                with open(os.path.join(folder, key + '.txt'), 'w') as out_file:
                    out_file.write(' '.join(val))


convert_json_to_txt()
