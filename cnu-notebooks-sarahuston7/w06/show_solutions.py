# Module to create widgets revealing exercise solutions.
#
# Example use in a notebook:
#
# from show_solutions import show
# show('Exercise 1', 'week1_solutions.md')
# show('Exercise 2', 'week1_solutions.md')
# 
# or:
#
# from show_solutions import show, initialise_path
# show = initialise_path(show, 'week1_solutions.md')
# show('Exercise 1')
# show('Exercise 2')


import re
import ipywidgets as widgets
from IPython.display import display, Markdown

def read_solution(tag, path):
    '''
    Reads solution from exercise 'tag', from file 'path'.
    Returns solution as string.
    '''
    try:
        # Retrieve solution code from script, as a string
        with open(path) as f:
            sol = ''
            write_line = False

            # Read line-by-line
            for l in f:
                if re.match(r'#+ ?' + tag, l):
                    # Found a header with the exercise tag,
                    # start writing at the next line
                    write_line = True
                    continue

                # Continue writing lines until end tag
                if write_line:
                    if l.startswith('---'):
                        # Reached the end tag, stop reading file
                        write_line = False
                        break
                    else:
                        # Append current line to the solution to display
                        sol += l
        return sol

    except FileNotFoundError:
        # If the file is not found, assume that solutions have not been released yet
        return 'Solutions not available!'


def show(tag, path):
    '''
    Displays solution to a particular exercise.
    
    Input:
    tag (str): string corresponding to the exercise tag in the solutions file
    path (str): path to the Markdown file containing solutions
    '''

    # Create output area for the solution
    sol_area = widgets.Output(layout={'border': '1px solid green'})

    # Create accordion
    acc = widgets.Accordion(children=[sol_area], selected_index=None)
    acc.set_title(0, 'Solution')

    # Read solutions from file
    sol = read_solution(tag, path)

    # Add full solution to the display area, and display the accordion widget
    sol_area.append_display_data(Markdown(data=sol))
    display(acc)


def initialise_path(show, path):
    '''
    Wrapper to initialise the `path` input once and for all,
    for instance if using the same solution file across an entire notebook.

    Example use: at the top of the notebook:
    from solutions import show
    show = initialise_path(show, 'week1_solutions.md')

    Then, show() can be called with only the exercise tag: instead of
    show('Exercise 1', 'week1_solutions.md')

    we can now use:
    show('Exercise 1')
    '''
    def wrapper(tag):
        out = show(tag, path)
        return out
    return wrapper
