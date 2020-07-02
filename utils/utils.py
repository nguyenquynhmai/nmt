import math


def get_lr_multiplier(step: int, warmup_steps: int) -> float:
    r"""Calculate the learning rate multiplier given current step and the number
    of warm-up steps. The learning rate schedule follows a linear warm-up and
    square-root decay.
    """
    multiplier = (min(1.0, step / warmup_steps) *
                  (1 / math.sqrt(max(step, warmup_steps))))
    return multiplier