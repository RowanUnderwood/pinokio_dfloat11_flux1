import os
import torch
from diffusers import FluxFillPipeline # Changed to FluxFillPipeline
from diffusers.utils import load_image # For loading example image/mask
from dfloat11 import DFloat11Model
from PIL import Image
import argparse
import requests # For fetching example images
from io import BytesIO # For handling image bytes

def download_image(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        return Image.open(BytesIO(response.content)).convert("RGB")
    except Exception as e:
        print(f"Failed to download {url}: {e}")
        # Fallback: create a simple placeholder image
        return Image.new("RGB", (512, 512), color = "gray")


def main(output_image_path_arg):
    print("Initializing FLUX.1-Fill-dev pipeline...")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")

    try:
        pipe = FluxFillPipeline.from_pretrained( # Changed
            "black-forest-labs/FLUX.1-Fill-dev", # Changed model name
            torch_dtype=torch.bfloat16,
        )
        print("FLUX.1-Fill-dev bfloat16 pipeline loaded.")

        # Optional: CPU offloading for the main pipeline
        # print("Enabling model CPU offload for the main pipeline (if VRAM is limited)...")
        # pipe.enable_model_cpu_offload()

        print("Loading DFloat11 compressed transformer for FLUX.1-Fill-dev-DF11...")
        DFloat11Model.from_pretrained(
            'DFloat11/FLUX.1-Fill-dev-DF11', # Changed DFloat11 model name
            bfloat16_model=pipe.transformer,
        )
        print("DFloat11 transformer loaded and integrated.")

        # Load example image and mask for inpainting/filling
        print("Loading example image and mask...")
        # Using URLs from a diffusers example, replace with your own if needed
        img_url = "https://huggingface.co/datasets/diffusers/diffusers-images-docs/resolve/main/ ইয়াহিয়ার.png"
        mask_url = "https://huggingface.co/datasets/diffusers/diffusers-images-docs/resolve/main/cup_mask.png"

        init_image = download_image(img_url).resize((1024, 1024))
        mask_image = download_image(mask_url).resize((1024, 1024))

        prompt = "A golden retriever wearing a party hat"
        print(f"Generating image for inpainting with prompt: '{prompt}'...")
        print("This may take a moment...")

        image = pipe(
            prompt=prompt,
            image=init_image,
            mask_image=mask_image,
            num_inference_steps=10, # Adjust as needed
            guidance_scale=0.0 # FLUX.1-Fill-dev also typically uses 0.0
        ).images[0]
        print("Image generation complete.")

        output_dir = os.path.dirname(output_image_path_arg)
        if output_dir and not os.path.exists(output_dir):
            print(f"Creating output directory: {output_dir}")
            os.makedirs(output_dir, exist_ok=True)

        image.save(output_image_path_arg)
        print(f"SUCCESS: Saved output to {output_image_path_arg}")

    except Exception as e:
        print(f"ERROR: An error occurred: {e}")
        import traceback
        traceback.print_exc()
        with open(os.path.join(os.path.dirname(output_image_path_arg), "error_fill.txt"), "w") as f:
            f.write(str(e))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DFloat11 FLUX.1-Fill-dev Example Script")
    parser.add_argument(
        "--output_image_path",
        type=str,
        default="flux_1_fill_dev_output.png",
        help="Full path to save the generated image."
    )
    args = parser.parse_args()
    main(args.output_image_path)