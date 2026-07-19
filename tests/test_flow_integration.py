"""Tests for flow computation and inter-frame warp integration."""

import os

import cv2
import numpy as np
import pytest
import torch
from PIL import Image

from vibewarp.flow.flow_utils import (
    InputPadder,
    _load_frame_for_flow,
    compute_flow,
    get_flow_and_cc,
    load_raft_model,
)


class MockRAFTModel(torch.nn.Module):
    """Mock RAFT model that returns synthetic flow."""

    def forward(self, frame1, frame2, num_flow_updates=20):
        B, C, H, W = frame1.shape
        # Return small constant flow (2 px right, 1 px down)
        flow = torch.zeros(B, 2, H, W, device=frame1.device)
        flow[:, 0, :, :] = 2.0  # horizontal
        flow[:, 1, :, :] = 1.0  # vertical
        return None, flow


@pytest.fixture
def mock_raft():
    return MockRAFTModel().eval()


@pytest.fixture
def two_frames(tmp_path):
    """Create two dummy frame images."""
    img1 = Image.fromarray(np.random.randint(0, 255, (64, 80, 3), dtype=np.uint8))
    img2 = Image.fromarray(np.random.randint(0, 255, (64, 80, 3), dtype=np.uint8))
    p1 = str(tmp_path / "000000.jpg")
    p2 = str(tmp_path / "000001.jpg")
    img1.save(p1)
    img2.save(p2)
    return p1, p2


class TestLoadFrameForFlow:
    def test_from_path(self, tmp_path):
        img = Image.fromarray(np.zeros((32, 48, 3), dtype=np.uint8))
        p = str(tmp_path / "test.png")
        img.save(p)
        t = _load_frame_for_flow(p, device='cpu')
        assert t.shape == (1, 3, 32, 48)

    def test_from_pil(self):
        img = Image.fromarray(np.zeros((32, 48, 3), dtype=np.uint8))
        t = _load_frame_for_flow(img, device='cpu')
        assert t.shape == (1, 3, 32, 48)

    def test_resize(self, tmp_path):
        img = Image.fromarray(np.zeros((100, 150, 3), dtype=np.uint8))
        p = str(tmp_path / "test.png")
        img.save(p)
        t = _load_frame_for_flow(p, size=(64, 48), device='cpu')
        assert t.shape == (1, 3, 48, 64)


class TestComputeFlow:
    def test_returns_numpy_hw2(self, mock_raft, tmp_path):
        img1 = Image.fromarray(np.zeros((64, 80, 3), dtype=np.uint8))
        img2 = Image.fromarray(np.zeros((64, 80, 3), dtype=np.uint8))
        f1 = _load_frame_for_flow(img1, device='cpu')
        f2 = _load_frame_for_flow(img2, device='cpu')
        flow = compute_flow(f1, f2, mock_raft, half=False)
        assert isinstance(flow, np.ndarray)
        assert flow.shape[2] == 2

    def test_accepts_string_paths(self, mock_raft, two_frames):
        p1, p2 = two_frames
        flow = compute_flow(p1, p2, mock_raft, half=False)
        assert isinstance(flow, np.ndarray)
        assert flow.shape[2] == 2

    def test_mock_flow_values(self, mock_raft):
        img = Image.fromarray(np.zeros((64, 80, 3), dtype=np.uint8))
        f1 = _load_frame_for_flow(img, device='cpu')
        f2 = _load_frame_for_flow(img, device='cpu')
        flow = compute_flow(f1, f2, mock_raft, half=False)
        # Mock returns constant (2, 1) flow — check after unpadding
        assert flow.shape[0] == 64
        assert flow.shape[1] == 80
        # Values should be close to (2, 1) from mock
        assert np.allclose(flow[..., 0], 2.0, atol=0.1)
        assert np.allclose(flow[..., 1], 1.0, atol=0.1)


class TestGetFlowAndCC:
    def test_computes_and_caches(self, mock_raft, two_frames, tmp_path):
        p1, p2 = two_frames
        flow_path = str(tmp_path / "flow" / "000000.npy")

        flow, cc = get_flow_and_cc(
            frame1_path=p1, frame2_path=p2,
            flow_path=flow_path,
            model=mock_raft,
        )
        assert isinstance(flow, np.ndarray)
        assert flow.shape[2] == 2
        assert os.path.exists(flow_path)

    def test_loads_cached_flow(self, two_frames, tmp_path):
        p1, p2 = two_frames
        flow_path = str(tmp_path / "cached.npy")
        # Pre-cache a flow file
        fake_flow = np.ones((64, 80, 2), dtype=np.float32) * 5.0
        np.save(flow_path, fake_flow)

        flow, cc = get_flow_and_cc(
            frame1_path=p1, frame2_path=p2,
            flow_path=flow_path,
            model=None,  # should not need model
        )
        np.testing.assert_array_equal(flow, fake_flow)

    def test_raises_without_model_and_no_cache(self, two_frames, tmp_path):
        p1, p2 = two_frames
        flow_path = str(tmp_path / "nonexistent.npy")
        with pytest.raises(ValueError, match="RAFT model required"):
            get_flow_and_cc(
                frame1_path=p1, frame2_path=p2,
                flow_path=flow_path,
                model=None,
            )

    def test_with_consistency_map(self, mock_raft, two_frames, tmp_path):
        p1, p2 = two_frames
        flow_path = str(tmp_path / "flow" / "000000.npy")
        cc_path = str(tmp_path / "flow" / "000000_cc.jpg")

        flow, cc = get_flow_and_cc(
            frame1_path=p1, frame2_path=p2,
            flow_path=flow_path,
            cc_path=cc_path,
            model=mock_raft,
        )
        assert cc is not None
        assert cc.shape[2] == 3  # 3-channel consistency map
        assert os.path.exists(cc_path)

    def test_notebook_uses_20_backward_and_12_forward_updates(
            self, monkeypatch, two_frames, tmp_path):
        import vibewarp.flow.flow_utils as flow_utils

        calls = []

        def fake_compute(frame1, frame2, model, half=False, num_flow_updates=20):
            calls.append(num_flow_updates)
            _, _, height, width = frame1.shape
            return np.zeros((height, width, 2), dtype=np.float32)

        monkeypatch.setattr(flow_utils, 'compute_flow', fake_compute)
        p1, p2 = two_frames
        get_flow_and_cc(
            frame1_path=p1,
            frame2_path=p2,
            flow_path=str(tmp_path / 'flow.npy'),
            cc_path=str(tmp_path / 'cc.png'),
            model=object(),
            num_flow_updates=20,
        )
        assert calls == [20, 12]

    def test_caches_unclamped_flow_but_returns_clamped_flow(
            self, monkeypatch, two_frames, tmp_path):
        import vibewarp.flow.flow_utils as flow_utils

        def tiny_flow(frame1, frame2, model, half=False, num_flow_updates=20):
            _, _, height, width = frame1.shape
            return np.full((height, width, 2), 0.1, dtype=np.float32)

        monkeypatch.setattr(flow_utils, 'compute_flow', tiny_flow)
        p1, p2 = two_frames
        flow_path = str(tmp_path / 'flow.npy')
        returned, _ = get_flow_and_cc(
            frame1_path=p1, frame2_path=p2,
            flow_path=flow_path, model=object(),
        )
        assert np.count_nonzero(returned) == 0
        assert np.all(np.load(flow_path) == np.float32(0.1))

    def test_force_recompute(self, mock_raft, two_frames, tmp_path):
        p1, p2 = two_frames
        flow_path = str(tmp_path / "flow" / "000000.npy")

        # First computation
        get_flow_and_cc(
            frame1_path=p1, frame2_path=p2,
            flow_path=flow_path, model=mock_raft,
        )
        mtime1 = os.path.getmtime(flow_path)

        # Force recompute
        import time
        time.sleep(0.05)
        get_flow_and_cc(
            frame1_path=p1, frame2_path=p2,
            flow_path=flow_path, model=mock_raft, force=True,
        )
        mtime2 = os.path.getmtime(flow_path)
        assert mtime2 > mtime1

    def test_with_flow_maxsize(self, mock_raft, tmp_path):
        # Create larger frames
        img1 = Image.fromarray(np.random.randint(0, 255, (200, 300, 3), dtype=np.uint8))
        img2 = Image.fromarray(np.random.randint(0, 255, (200, 300, 3), dtype=np.uint8))
        p1 = str(tmp_path / "big1.jpg")
        p2 = str(tmp_path / "big2.jpg")
        img1.save(p1)
        img2.save(p2)
        flow_path = str(tmp_path / "flow" / "big.npy")

        flow, _ = get_flow_and_cc(
            frame1_path=p1, frame2_path=p2,
            flow_path=flow_path, model=mock_raft,
            flow_maxsize=128,
        )
        # Flow should be at computation resolution (not original),
        # matching notebook behavior — caller resizes at warp time
        assert max(flow.shape[0], flow.shape[1]) <= 128


class TestLoadRaftModel:
    def test_function_exists(self):
        """load_raft_model should be importable and callable."""
        assert callable(load_raft_model)

    def test_signature(self):
        import inspect
        sig = inspect.signature(load_raft_model)
        params = list(sig.parameters.keys())
        assert 'half_precision' in params
        assert 'device' in params
