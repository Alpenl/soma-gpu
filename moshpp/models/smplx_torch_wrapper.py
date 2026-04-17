from dataclasses import dataclass

import torch


@dataclass
class TorchBodyOutput:
    vertices: torch.Tensor
    joints: torch.Tensor


def _broadcast_batch(tensor, batch_size):
    if tensor is None:
        return None
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    if tensor.shape[0] == batch_size:
        return tensor
    if tensor.shape[0] == 1:
        return tensor.expand(batch_size, -1).clone()
    raise ValueError(f"Cannot broadcast tensor with shape {tuple(tensor.shape)} to batch size {batch_size}")


class SmplxTorchWrapper:
    def __init__(self, body_model=None, body_model_factory=None, surface_model_type="smplx"):
        if surface_model_type != "smplx":
            raise NotImplementedError(f"Unsupported surface_model_type: {surface_model_type}")
        if body_model is None and body_model_factory is None:
            raise ValueError("Provide body_model or body_model_factory.")
        if body_model is not None and body_model_factory is not None:
            raise ValueError("Provide only one of body_model or body_model_factory.")
        self.body_model = body_model
        self.body_model_factory = body_model_factory
        self.surface_model_type = surface_model_type

    def __call__(self, fullpose, betas, transl, expression=None):
        if fullpose.ndim != 2 or fullpose.shape[1] != 165:
            raise ValueError(f"fullpose must have shape (B, 165), got {tuple(fullpose.shape)}")

        if self.body_model is None:
            self.body_model = self.body_model_factory()

        batch_size = fullpose.shape[0]
        betas = _broadcast_batch(betas, batch_size)
        transl = _broadcast_batch(transl, batch_size)
        expression = _broadcast_batch(expression, batch_size)

        kwargs = dict(
            global_orient=fullpose[:, 0:3],
            body_pose=fullpose[:, 3:66],
            jaw_pose=fullpose[:, 66:69],
            leye_pose=fullpose[:, 69:72],
            reye_pose=fullpose[:, 72:75],
            left_hand_pose=fullpose[:, 75:120],
            right_hand_pose=fullpose[:, 120:165],
            betas=betas,
            transl=transl,
        )
        if expression is not None:
            kwargs["expression"] = expression

        model_output = self.body_model(**kwargs)

        return TorchBodyOutput(vertices=model_output.vertices, joints=model_output.joints)
