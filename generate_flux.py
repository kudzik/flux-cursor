import torch
from diffusers import FluxPipeline

PROMPT = "A dachshund riding a bicycle, cinematic, sharp, natural lighting"

if __name__ == "__main__":
    # Wybór precyzyjnego typu danych — float16 działa świetnie na RTX 4080 SUPER
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-schnell",
        torch_dtype=torch.float16
    ).to("cuda") # jawne przeniesienie na GPU

    # Jeśli chcesz oszczędzać VRAM przy większych rozdzielczościach:
    # pipe.enable_model_cpu_offload()

    # Generator z ustalonym seedem dla powtarzalności
    generator = torch.Generator(device="cuda").manual_seed(42)

    # Generacja obrazu
    image = pipe(
        prompt=PROMPT,
        guidance_scale=0.0,
        num_inference_steps=4,
        max_sequence_length=256,
        height=1344,  # proporcja 4:5
        width=1088,
        generator=generator,
    ).images[0]

    # Zapis obrazu
    image.save("dachshund_bike_1080x1350.png")
