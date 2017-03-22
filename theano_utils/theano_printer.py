"""
This has a static object, which can be used to print theano variables.
Call the print_me_this() on your variable, and it will be printed during training.
"""
import theano
import theano.tensor as T

_stuff_to_print = []

def print_me_this(text, theano_variable):
    _stuff_to_print.append(theano.printing.Print(text+":")(theano_variable))

def get_the_stuff_to_print():
    return _stuff_to_print