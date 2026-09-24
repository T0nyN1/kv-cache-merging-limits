TASK_REGISTRY = {}


def register_task(name: str):
    def decorator(cls):
        TASK_REGISTRY[name] = cls
        return cls

    return decorator


def get_evaluator(name: str):
    if name not in TASK_REGISTRY:
        raise ValueError(f"Task '{name}' not found. Available tasks: {list(TASK_REGISTRY.keys())}")
    return TASK_REGISTRY[name]
