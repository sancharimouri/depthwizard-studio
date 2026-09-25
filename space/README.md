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
short_description: DAv2-Small relative depth on CPU (API for DepthWizard2)
---

CPU-only inference endpoint for the Depth Wizard 2 (SIH26175) backend. It runs
[Depth-Anything-V2-Small](https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf)
(Apache-2.0) and returns **relative** depth, not elevation.

```python
from gradio_client import Client, handle_file
r = Client("sancharimouri/DepthWizard2").predict(handle_file("image.jpg"), api_name="/predict")
# r["shape"] == [518, 518]; depth = np.frombuffer(base64.b64decode(r["data_b64"]), "<f4").reshape(r["shape"])
```

This Space is on free CPU Basic hardware, which sleeps after 48 h idle, so the first call after a sleep is slow.
