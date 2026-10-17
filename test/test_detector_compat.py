"""Keep the deployed Frigate model request and inference contracts intact."""

import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "detector"))
from zmq_onnx_client import ZmqOnnxClient


class DetectorCompatTest(unittest.TestCase):
    def test_repeated_model_request_reuses_session(self):
        client = ZmqOnnxClient.__new__(ZmqOnnxClient)
        client.current_model = "yolov9-fixed.onnx"
        client.model_ready = True
        client.session = object()
        client._check_model_exists = lambda name: name == client.current_model
        client._load_model = lambda name: self.fail("reloaded an active model")

        reply = client._handle_model_request({"model_name": client.current_model})

        self.assertTrue(json.loads(reply[0])["model_loaded"])

    def test_ssd_header_keeps_yolo_detections(self):
        raw = np.zeros((1, 84, 85), dtype=np.float32)
        raw[0, :5, 0] = [160, 160, 40, 40, 0.95]
        session = SimpleNamespace(
            get_inputs=lambda: [SimpleNamespace(name="images")],
            run=lambda outputs, inputs: [raw],
        )
        client = ZmqOnnxClient.__new__(ZmqOnnxClient)
        client.session = session
        client.zero_result = np.zeros((20, 6), dtype=np.float32)

        result = client._run_inference(
            np.zeros((1, 3, 320, 320), dtype=np.float32),
            {"model_type": "ssd", "width": 320, "height": 320},
        )

        np.testing.assert_allclose(
            result[0], [0, 0.95, 0.4375, 0.4375, 0.5625, 0.5625], rtol=1e-6
        )


if __name__ == "__main__":
    unittest.main()
