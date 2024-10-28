
# Beginning of complexity.py
def choose_model_based_on_complexity(complexity_factor):
    if complexity_factor < 10:
        return SimpleModel()
    elif 10 <= complexity_factor < 100:
        return MediumModel()
    else:
        return ComplexModel()

def choose_evaluation_metric(complexity_factor):
    if complexity_factor < 10:
        return mean_squared_error
    elif 10 <= complexity_factor < 100:
        return mean_absolute_error
    else:
        return partial(mean_squared_error, squared=False)
# End of complexity.py
