import functools

@functools.lru_cache(maxsize=1024)
def get_complexity_factor(input_data):
    try:
        complexity_factor = len(input_data)
        return complexity_factor
    except Exception as e:
        logging.error(f"Error in get_complexity_factor: {str(e)}")
        return 0

def choose_num_iterations_based_on_complexity(complexity_factor):
    if complexity_factor < 10:
        return 50
    elif 10 <= complexity_factor < 100:
        return 100
    else:
        return 200