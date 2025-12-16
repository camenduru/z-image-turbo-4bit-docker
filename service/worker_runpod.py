import os, shutil, random, time, uuid, boto3, runpod
from pathlib import Path
from datetime import datetime

import torch
import diffusers
from sdnq import SDNQConfig

pipe = diffusers.ZImagePipeline.from_pretrained("/content/z-image-4bit", torch_dtype=torch.bfloat16, device_map="cuda")
pipe.reset_device_map()
pipe.enable_model_cpu_offload()

@torch.inference_mode()
def generate(input):
    try:
        tmp_dir="/content/output"
        os.makedirs(tmp_dir, exist_ok=True)
        unique_id = uuid.uuid4().hex[:6]
        current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        s3_access_key_id = os.getenv('s3_access_key_id')
        s3_secret_access_key = os.getenv('s3_secret_access_key')
        s3_endpoint_url = os.getenv('s3_endpoint_url')
        s3_region_name = os.getenv('s3_region_name')
        s3_bucket_name = os.getenv('s3_bucket_name')
        s3_bucket_folder = os.getenv('s3_bucket_folder')
        s3 = boto3.client('s3', aws_access_key_id=s3_access_key_id, aws_secret_access_key=s3_secret_access_key, endpoint_url=s3_endpoint_url, region_name=s3_region_name)

        values = input["input"]
        job_id = values['job_id']

        positive_prompt = values['positive_prompt']
        negative_prompt = values['negative_prompt']
        seed = values['seed'] # 0
        steps = values['steps'] # 9
        cfg = values['cfg'] # 0.0
        width = values['width'] # 1024
        height = values['height'] # 1024

        if seed == 0:
            random.seed(int(time.time()))
            seed = random.randint(0, 18446744073709551615)

        image = pipe(prompt=positive_prompt, negative_prompt=negative_prompt, height=height,width=width, num_inference_steps=steps, guidance_scale=cfg, generator=torch.Generator("cuda").manual_seed(seed),).images[0]
        image.save(f"{tmp_dir}/z_image_turbo.png")

        result = f"{tmp_dir}/z_image_turbo.png"
        
        s3_key =  f"{s3_bucket_folder}/z_image_turbo-{current_time}-{seed}-{unique_id}.png"
        s3.upload_file(result, s3_bucket_name, s3_key, ExtraArgs={'ContentType': 'image/png'})
        result_url = f"{s3_endpoint_url}/{s3_bucket_name}/{s3_key}"

        return {"job_id": job_id, "result": result_url, "status": "DONE"}
    except Exception as e:
        return {"job_id": job_id, "result": str(e), "status": "FAILED"}
    finally:
        directory_path = Path(tmp_dir)
        if directory_path.exists():
            shutil.rmtree(directory_path)
            print(f"Directory {directory_path} has been removed successfully.")
        else:
            print(f"Directory {directory_path} does not exist.")

runpod.serverless.start({"handler": generate})