"""
MinkowskiEngine backend wrapper.

This module is intentionally isolated.
If MinkowskiEngine is not installed, importing this file raises ImportError
and the pipeline falls back gracefully to PrototypeSparseBackend.

Future: replace the stub forward() with a real MinkUNet checkpoint.
"""
from __future__ import annotations


class MinkowskiBackend:
    """
    Wrapper for a MinkowskiEngine-based sparse semantic network.

    PROTOTYPE NOTE: Currently a stub. The interface is ready for
    a real MinkUNet / SPVCNN checkpoint.
    Future: load a pretrained .pth checkpoint here.
    """
    name = "MinkowskiEngine (sparse convolution)"

    def __init__(self):
        # This raises ImportError if ME is not installed — caught by sparse_backend.py
        import MinkowskiEngine  # noqa: F401
        self._me = MinkowskiEngine
        # TODO: load pretrained checkpoint
        # self.model = MinkUNet(...)
        # self.model.load_state_dict(torch.load("path/to/checkpoint.pth"))

    def predict_tile(self, points, ground_mask, height_mean,
                     height_variance, intensity_mean, voxel_size=0.05):
        """Stub — replace with real ME inference."""
        raise NotImplementedError(
            "MinkowskiBackend.predict_tile() needs a real checkpoint. "
            "The system should have fallen back to PrototypeSparseBackend."
        )
