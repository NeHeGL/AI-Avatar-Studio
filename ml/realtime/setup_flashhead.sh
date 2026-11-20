set -e
cd "$(dirname "$0")"; test -d flashhead || git clone --depth 1 https://github.com/Soul-AILab/SoulX-FlashHead.git flashhead; cd flashhead; PY=".venv/Scripts/python.exe"; test -d .venv || uv venv .venv --python 3.11
uv pip install --python $PY torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128 2>&1 | tail -2
grep -v -i "nccl\|xformers\|gradio\|xfuser\|mediapipe\|decord" requirements.txt > req_win.txt
uv pip install --python $PY -r req_win.txt huggingface_hub psutil 2>&1 | tail -3
uv pip install --python $PY xfuser 2>&1 | tail -3 || echo "xfuser install failed (will stub)"
uv pip install --python $PY decord mediapipe 2>&1 | tail -2 || echo "decord/mediapipe failed"
$PY -c "import huggingface_hub as h; h.snapshot_download('Soul-AILab/SoulX-FlashHead-1_3B', local_dir='models/SoulX-FlashHead-1_3B', allow_patterns=['Model_Lite/*','*.json','*.md','*.txt'])"
$PY -c "import huggingface_hub as h; h.snapshot_download('facebook/wav2vec2-base-960h', local_dir='models/wav2vec2-base-960h', allow_patterns=['*.json','*.txt','*.safetensors','*.bin'])"
echo SETUP_DONE
