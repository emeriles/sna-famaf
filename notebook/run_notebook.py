import codecs

import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor
from os.path import join
import time


def generate_notebook(notebook_path, dataset_path, output_path):
    notebook_name = notebook_path
    out_nb_fname = output_path
    print("Computing %s" % out_nb_fname)
    nb = nbf.read(open(notebook_name), as_version=4)
    nb['cells'][0]['source'] = 'csv_path ="{}"'.format(dataset_path)
    ep = ExecutePreprocessor(timeout=None, kernel_name='python3')
    ep.preprocess(nb, {'metadata': {'path': './'}})
    with codecs.open(out_nb_fname, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)


def run_notebook(notebook_path, input_csv, output_path):
    dataset_path = input_csv
    start_time = time.time()
    generate_notebook(notebook_path=notebook_path, dataset_path=dataset_path, output_path=output_path)
    run_time = time.time() - start_time
    print('Time to run {} was {} secs.'.format(dataset_path, run_time))
