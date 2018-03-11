### A module for storing global variables (which are used in multiple files)

def init():
    # ref: https://stackoverflow.com/questions/13034496/using-global-variables-between-files
    # Global variables for length of inputs & outputs
    global PROBLEM_LENGTH
    global TEMPLATE_LENGTH
    global SEED
    PROBLEM_LENGTH = 105
    TEMPLATE_LENGTH = 30
    SEED = 23