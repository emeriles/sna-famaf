import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Do all kinds of stuff with the project')

    action_choices = [
        'run_notebook',
    ]
    parser.add_argument('action', metavar='ACTION', type=str,
                        help='action to be performed. One of {}'.format(action_choices), choices=action_choices)

    parser.add_argument('-n', metavar='NOTEBOOK_PATH', type=str,
                        help='path for notebook to run', required=True)
    parser.add_argument('-i', metavar='INPUT', type=str,
                        help='path of csv as input', required=True)
    parser.add_argument('-o', metavar='OUTPUT', type=str,
                        help='path of .ipynb as output', required=True)

    args = parser.parse_args()

    print(args.n)
    print(args.i)
    print(args.o)
    print(args.action)

    print('RUNNING {} with args: '.format(args.action))
    print('\tInput notebook:' + args.n)
    print('\tInput csv: ' + args.i)
    print('\tNotebook output ' + args.o)
    print('\t' + args.action)
    if args.action == 'run_notebook':
        from run_notebook import run_notebook
        run_notebook(notebook_path=args.n, input_csv=args.i, output_path=args.o)
