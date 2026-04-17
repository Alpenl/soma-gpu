def choose_stageii_backend(backend_name, chumpy_backend, torch_backend):
    if backend_name in (None, "chumpy"):
        return chumpy_backend
    if backend_name == "torch":
        return torch_backend
    raise ValueError(f"Unsupported stageii backend: {backend_name}")


def load_stageii_backend(backend_name, chumpy_backend, load_torch_backend):
    if backend_name in (None, "chumpy"):
        return chumpy_backend
    if backend_name == "torch":
        return load_torch_backend()
    raise ValueError(f"Unsupported stageii backend: {backend_name}")
