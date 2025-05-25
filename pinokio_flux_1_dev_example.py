import os
import torch
from diffusers import FluxPipeline
from dfloat11 import DFloat11Model # This is the pip installed library
from PIL import Image
import argparse

def main(output_image_path_arg):
    print("Initializing FLUX.1-dev pipeline...")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")

    try:
        pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            torch_dtype=torch.bfloat16,
            # For low VRAM, consider uncommenting and adjusting:
            # variant="fp16", # May require FP16 model, DFloat11 example uses BF16
            # low_cpu_mem_usage=False, # Set to True if memory is very tight
            # device_map="auto", # Offloads to CPU/Disk if VRAM is insufficient
        )
        print("FLUX.1-dev bfloat16 pipeline loaded.")

        # To enable CPU offloading for the main pipeline (if not using device_map="auto" above):
        # print("Enabling model CPU offload for the main pipeline (if VRAM is limited)...")
        # pipe.enable_model_cpu_offload()


        print("Loading DFloat11 compressed transformer for FLUX.1-dev-DF11...")
        # DFloat11Model.from_pretrained will download the model from Hugging Face Hub
        # on first run and cache it.
        # It replaces the transformer in `pipe` directly.
        DFloat11Model.from_pretrained(
            'DFloat11/FLUX.1-dev-DF11',
            bfloat16_model=pipe.transformer,
            # If you used device_map="auto" or offloaded the bfloat16_model to CPU,
            # you might need to specify device="cpu" for DFloat11Model too.
            # Example: device="cpu" if pipe.device.type == 'cpu' else None
            # By default, it tries to load to the same device as bfloat16_model or GPU.
        )
        print("DFloat11 transformer loaded and integrated.")

        prompt = "A majestic cat king on a throne, digital art, highly detailed, intricate, fantasy"
        print(f"Generating image for prompt: '{prompt}'...")
        print("This may take a moment, especially on the first run (model download) and depending on your GPU.")

        # Generation
        image = pipe(
            prompt=prompt,
            num_inference_steps=10, # FLUX models are fast, 8-12 steps often enough
            guidance_scale=0.0 # FLUX.1-dev is trained without guidance
        ).images[0]
        print("Image generation complete.")

        # Ensure output directory exists
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
        # Create a dummy error file for Pinokio to potentially detect failure
        with open(os.path.join(os.path.dirname(output_image_path_arg), "error_dev.txt"), "w") as f:
            f.write(str(e))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DFloat11 FLUX.1-dev Example Script")
    parser.add_argument(
        "--output_image_path",
        type=str,
        default="flux_1_dev_output.png",
        help="Full path to save the generated image."
    )
    args = parser.parse_args()
    main(args.output_image_path)