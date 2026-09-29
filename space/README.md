---
title: DepthWizard2
emoji: 🏔️
colorFrom: green
colorTo: gray
sdk: gradio
sdk_version: 6.28.0
python_version: '3.11'
app_file: app.py
pinned: false
license: apache-2.0
short_description: DAv2-Small relative depth on ZeroGPU (API for DepthWizard2)
---

ZeroGPU (`zero-a10g`) inference endpoint for the Depth Wizard 2 (SIH26175) backend. It runs
[Depth-Anything-V2-Small](https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf)
(Apache-2.0) and returns **relative** depth, not elevation.

```python
from gradio_client import Client, handle_file
r = Client("sancharimouri/DepthWizard2").predict(handle_file("image.jpg"), api_name="/predict")
# r["encoding"] == "u16-zlib", r["shape"] == [518, 518]
# q = np.frombuffer(zlib.decompress(base64.b64decode(r["data_b64"])), "<u2").reshape(r["shape"])
# depth = r["min"] + q / 65535 * (r["max"] - r["min"])
```

This Space runs on ZeroGPU (`zero-a10g`). Each call draws on the caller's daily ZeroGPU quota, and the Space sleeps after
48 h idle, so the first call after a sleep is slow.
