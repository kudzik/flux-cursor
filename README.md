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

### 2) Wybór wariantu PyTorch (GPU/CPU)
Wybierz jedną z opcji:

- GPU (CUDA 12.1, rekomendowane dla RTX):
  ```powershell
  pip install -r requirements-cu121.txt
  ```
- CPU (bez CUDA, wolniejsze generowanie):
  ```powershell
  pip install -r requirements-cpu.txt
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

## Wersje i zgodność
- Torch: `2.6.0` (GPU CUDA 12.1 i CPU warianty w osobnych plikach wymagań)
- TorchVision: `0.21.0`, Torchaudio: `2.6.0`
- Diffusers: `0.35.1`, Transformers: `4.45.2`

Na innych wersjach CUDA (np. 12.4) zadziała po zmianie indeksu i wersji torch – przygotowaliśmy gotowe pliki dla CUDA 12.1 i CPU, bo w tych konfiguracjach testowo działa akceleracja GPU bez konfliktów.

## Licencja modelu
Model `FLUX.1-schnell` jest dostępny na licencji Apache-2.0 – szczegóły na stronie modelu.
