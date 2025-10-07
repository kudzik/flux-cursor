# FLUX.1-schnell – generacja obrazu (Python, CUDA)

Ten projekt demonstruje lokalną generację obrazu modelem `black-forest-labs/FLUX.1-schnell` z użyciem biblioteki Diffusers na GPU (CUDA).

Źródło modelu: [Hugging Face – FLUX.1-schnell](https://huggingface.co/black-forest-labs/FLUX.1-schnell)

## Wymagania
- Windows 10/11
- NVIDIA GPU (np. RTX 4080 Super) + sterowniki CUDA
- Python 3.11–3.13
- Konto Hugging Face i akceptacja warunków dostępu do modelu (na stronie modelu)

## Szybki start
### 1) Virtualenv
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2) PyTorch z CUDA
Zainstaluj PyTorch dopasowany do Twojej wersji CUDA (dla CUDA 12.4):
```powershell
pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision torchaudio
```

### 3) Zależności projektu
```powershell
pip install -r requirements.txt
```

### 4) Hugging Face – dostęp do modelu
- Zaloguj się i zaakceptuj warunki na stronie modelu.
- (Opcjonalnie) zaloguj się CLI: `huggingface-cli login`

### 5) Generacja (CUDA)
```powershell
python generate_flux.py
```
Domyślnie skrypt generuje obraz w proporcjach 4:5 (dobranych do wielokrotności 16) i zapisuje plik `dachshund_bike_1080x1350.png`.

## Dostosowanie
- Prompt ustawisz w `generate_flux.py` w stałej `PROMPT`.
- Rozdzielczość: zmień `height`/`width` (zalecane wielokrotności 16). Dla wiernego 4:5 polecam `width=1088`, `height=1360`.
- Szybkość vs VRAM:
  - Maks. szybkość: trzymaj całość na GPU (bez `enable_model_cpu_offload()`).
  - Mniejsze zużycie VRAM: włącz `pipe.enable_model_cpu_offload()`.
- Jakość: zwiększ `num_inference_steps` (np. 8), dodaj `negative_prompt`.

## Rozwiązywanie problemów
- Konflikt `offload_state_dict`: używamy `transformers==4.45.2` i `diffusers==0.35.1` (zob. `requirements.txt`).
- Tokenizer T5: doinstaluj `sentencepiece` oraz `protobuf` (są w `requirements.txt`).
- OOM CUDA: zmniejsz rozdzielczość lub włącz `enable_model_cpu_offload()`.

## Licencja modelu
Model `FLUX.1-schnell` jest dostępny na licencji Apache-2.0 – szczegóły na stronie modelu.
