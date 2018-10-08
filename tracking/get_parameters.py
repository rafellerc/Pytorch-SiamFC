import json
from collections import namedtuple
from os.path import abspath, join, dirname

ROOT_DIR = dirname(dirname(abspath(__file__)))


def get_parameters(exp_name='default'):
    """ Parses the parameters contained in the json files in
        root/tracking/experiments/<experiment>
    Args:
        exp_name (string): The name of the experiment being performed, it is
        the basename of folder containing the parameter files.

    Returns:
        hp (named_tuple): The tuple containing the hyperparameters.
        evaluation (named_tuple): The tuple containing the evaluation
        parameters such as the sequences being evaluated.
        run (named_tuple): The tuple containing the run information, such as
        if the user requests visualization or video-making.
        env (named_tuple): The tuple containing the environment information,
        such as the folder containing the data.
        design (named_tuple): The tuple containing the design and architecture
        information, such as the network being used.
    """

    with open(join(ROOT_DIR, 'tracking', 'experiments',  exp_name,
                   'hyperparams.json')) as json_file:
        hp = json.load(json_file)
    with open(join(ROOT_DIR, 'tracking', 'experiments', exp_name,
                   'evaluation.json')) as json_file:
        evaluation = json.load(json_file)
    with open(join(ROOT_DIR, 'tracking', 'experiments', exp_name,
                   'run.json')) as json_file:
        run = json.load(json_file)
    with open(join(ROOT_DIR, 'tracking', 'experiments', exp_name,
                   'environment.json')) as json_file:
        env = json.load(json_file)
    with open(join(ROOT_DIR, 'tracking', 'experiments', exp_name,
                   'design.json')) as json_file:
        design = json.load(json_file)

    hp = namedtuple('hp', hp.keys())(**hp)
    evaluation = namedtuple('evaluation', evaluation.keys())(**evaluation)
    run = namedtuple('run', run.keys())(**run)
    env = namedtuple('env', env.keys())(**env)
    design = namedtuple('design', design.keys())(**design)

    return hp, evaluation, run, env, design
