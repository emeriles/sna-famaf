import codecs

import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor
from os.path import join


def generate_notebook(dataset_path):
    notebook_name = 'Exploratory_visualization_corrected.ipynb'
    out_nb_fname = join('./notebooks_outputs/', ' '.join([notebook_name]))
    print("Computing %s" % out_nb_fname)
    nb = nbf.read(open(notebook_name), as_version=4)
    nb['cells'][0]['source'] = 'csv_path ="{}"'.format(dataset_path)
    ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
    ep.preprocess(nb, {'metadata': {'path': './'}})
    with codecs.open(out_nb_fname, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)


if __name__ == '__main__':
    dataset_paths = ['../../database/dayli_collections/dayli_col.csv']
    for dataset_path in dataset_paths:
        print("Running for {}".format(dataset_path))
        generate_notebook(dataset_path=dataset_path)
