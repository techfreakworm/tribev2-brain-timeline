import types
import numpy as np
from tribescore import inference


def test_run_inference_flags_no_speech(monkeypatch):
    # Stub a model whose events have NO Word rows; assert out_info["no_speech"] is set.
    import pandas as pd
    events = pd.DataFrame({"type": ["Audio", "Video"], "filepath": ["a", "v"]})

    class _Model:
        def get_events_dataframe(self, **kw):
            return events
        def predict(self, ev):
            seg = types.SimpleNamespace(start=0.0)
            return np.zeros((1, 20484)), [seg]

    monkeypatch.setattr(inference, "assert_model_runtime", lambda: None)
    info = {}
    inference.run_inference(_Model(), "video", "v.mp4", audio_only=False, out_info=info)
    assert info.get("no_speech") is True


def test_run_inference_no_flag_when_words_present(monkeypatch):
    import pandas as pd
    events = pd.DataFrame({"type": ["Audio", "Word"], "filepath": ["a", "a"],
                           "text": [None, "hi"]})

    class _Model:
        def get_events_dataframe(self, **kw):
            return events
        def predict(self, ev):
            seg = types.SimpleNamespace(start=0.0)
            return np.zeros((1, 20484)), [seg]

    monkeypatch.setattr(inference, "assert_model_runtime", lambda: None)
    info = {}
    inference.run_inference(_Model(), "video", "v.mp4", audio_only=False, out_info=info)
    assert "no_speech" not in info
