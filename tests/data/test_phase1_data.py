import pytest

from visionaid.io.sources import VideoFileSource

pytestmark = pytest.mark.data


def test_read_full_video(video_path, cfg):
    src = VideoFileSource.from_config(cfg)
    n = 0
    last_t = -1.0
    while (f := src.read()) is not None:
        assert f.image.shape[1] == cfg.video.width
        assert f.t > last_t
        last_t = f.t
        n += 1
    assert n == src.frame_count
    assert f is None
    # portrait video after rotation
    assert src.out_h > src.out_w
