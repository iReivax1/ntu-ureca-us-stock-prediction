import datetime
import functools
import warnings


def log_time_taken(func):

    def func_wrapper(*args, **kwargs):
        # log start time
        dt_started = datetime.datetime.now()

        func_result = func(*args, **kwargs)

        # log end time
        dt_ended = datetime.datetime.now()
        print(dt_started, dt_ended, (dt_ended - dt_started))

        return func_result

    return func_wrapper


def deprecated(func):
    """This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emmitted
    when the function is used."""

    @functools.wraps(func)
    def new_func(*args, **kwargs):
        warnings.simplefilter('always', DeprecationWarning)     # turn off filter
        warnings.warn("Call to deprecated function {}.".format(func.__name__), category=DeprecationWarning, stacklevel=2)
        warnings.simplefilter('default', DeprecationWarning)    # reset filter
        return func(*args, **kwargs)

    return new_func
