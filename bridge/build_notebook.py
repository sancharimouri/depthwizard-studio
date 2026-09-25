"""Regenerate bridge/dav2_colab_bridge.ipynb from bridge/dav2_server.py (keeps them identical)."""
import json
from pathlib import Path

HERE = Path(__file__).parent
server = (HERE / "dav2_server.py").read_text()
SAMPLE = "https://github.com/sancharimouri/depthwizard2-assets/releases/download/library-v1/sentinel2-almora__preview.jpg"


def md(s): return {"cell_type": "markdown", "metadata": {}, "source": s}
def code(s): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s}


cells = [
    md("# DepthWizard2 — TEMPORARY DAv2-Small bridge (Colab GPU + Cloudflare quick tunnel)\n\n"
       "**Temporary.** The tunnel URL dies when this notebook disconnects, the runtime is recycled, or the session ends. "
       "Keep this tab open and connected. Every restart gives a **new** URL, which must be set again as `DAV2_INFERENCE_URL` "
       "on the backend. No Cloudflare account or card is involved (quick tunnel). See `docs/DEPLOY.md` in the repo.\n\n"
       "Runtime → Change runtime type → **T4 GPU** (or better), then Runtime → Run all."),
    code("import torch, subprocess\n"
         "print(subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True).stdout)\n"
         "assert torch.cuda.is_available(), 'No GPU: Runtime > Change runtime type > GPU'\n"
         "print('torch', torch.__version__)"),
    code("# transformers pinned to the version the project runs (DPTImageProcessorPil); Colab's own torch/CUDA is kept\n"
         "!pip -q install transformers==5.17.0 python-multipart fastapi uvicorn"),
    code("%%writefile dav2_server.py\n" + server),
    code("import subprocess, time, urllib.request, json\n"
         "srv = subprocess.Popen(['python', '-m', 'uvicorn', 'dav2_server:app', '--host', '127.0.0.1', '--port', '8000'],\n"
         "                       stdout=open('server.log', 'w'), stderr=subprocess.STDOUT)\n"
         "for _ in range(180):\n"
         "    try:\n"
         "        print(json.load(urllib.request.urlopen('http://127.0.0.1:8000/health'))); break\n"
         "    except Exception:\n"
         "        assert srv.poll() is None, open('server.log').read()\n"
         "        time.sleep(1)\n"
         "else:\n"
         "    raise RuntimeError(open('server.log').read())"),
    code("# cloudflared: the official static Linux binary (no account, no login for a quick tunnel)\n"
         "!wget -q -O cloudflared https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 && chmod +x cloudflared\n"
         "# Logs go to a file, not an undrained pipe (a full pipe could block cloudflared on a log write).\n"
         "# Re-running this cell replaces the tunnel (NEW URL). Stopping the notebook kills it (Cloudflare error 1033).\n"
         "import re\n"
         "!pkill -f 'cloudflared tunnel' || true\n"
         "tun = subprocess.Popen(['./cloudflared', 'tunnel', '--no-autoupdate', '--url', 'http://127.0.0.1:8000'],\n"
         "                       stdout=open('cloudflared.log', 'w'), stderr=subprocess.STDOUT)\n"
         "URL = None\n"
         "for _ in range(60):\n"
         "    m = re.search(r'https://[a-z0-9-]+\\.trycloudflare\\.com', open('cloudflared.log').read())\n"
         "    if m:\n"
         "        URL = m.group(0); break\n"
         "    assert tun.poll() is None, open('cloudflared.log').read()\n"
         "    time.sleep(1)\n"
         "print('\\nDAV2_INFERENCE_URL =', URL)"),
    code("# Real end-to-end check through the public tunnel: a real Sentinel-2 preview -> 518x518 depth\n"
         "import base64, io, numpy as np, requests\n"
         f"img = requests.get('{SAMPLE}', timeout=60).content\n"
         "for _ in range(30):  # DNS for a fresh trycloudflare name can take a few seconds\n"
         "    try:\n"
         "        r = requests.post(URL + '/predict', files={'file': ('almora.jpg', img, 'image/jpeg')}, timeout=120); break\n"
         "    except requests.ConnectionError:\n"
         "        time.sleep(2)\n"
         "r.raise_for_status(); d = r.json()\n"
         "depth = np.frombuffer(base64.b64decode(d['data_b64']), '<f4').reshape(d['shape'])\n"
         "print(d['device'], d['shape'], 'infer', d['infer_s'], 's  range', round(float(depth.min()), 3), '..', round(float(depth.max()), 3),\n"
         "      ' std', round(float(depth.std()), 3))\n"
         "assert depth.shape == (518, 518) and np.isfinite(depth).all() and depth.std() > 0"),
    code("# Keep-alive monitor: leave this running. It only reports; it cannot keep Colab alive if the tab closes.\n"
         "while True:\n"
         "    try:\n"
         "        print(time.strftime('%H:%M:%S'), URL, 'cloudflared', 'running' if tun.poll() is None else f'EXITED {tun.returncode}',\n"
         "              requests.get(URL + '/health', timeout=30).json())\n"
         "    except Exception as e:\n"
         "        print(time.strftime('%H:%M:%S'), 'tunnel check failed:', e)\n"
         "    time.sleep(60)"),
]
for c in cells:  # notebook format stores source as a list of lines
    c["source"] = c["source"].splitlines(keepends=True)
nb = {"nbformat": 4, "nbformat_minor": 5, "cells": cells,
      "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
                   "kernelspec": {"name": "python3", "display_name": "Python 3"}}}
(HERE / "dav2_colab_bridge.ipynb").write_text(json.dumps(nb, indent=1) + "\n")
print("wrote bridge/dav2_colab_bridge.ipynb")
