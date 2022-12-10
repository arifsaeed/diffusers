from diffusers import StableDiffusionPipeline
import torch

model_id = "/home/arif/Documents/design/sandpit/trainedsd/output/checkpoint-500"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")

prompts = ["A cartoon image of wunzag","A cartoon image of extofer","A cartoon image of gwala","A cartoon image of rawdib","A image of renwa","A cartoon image of riglel","A cartoon image of ungera","A cartoon image of wunzag"]
prompt_names=['wunzag','extofer','gwala','rawdib','renwa','riglel','ungera','wunzag']

for idx,prompt in enumerate(prompts):
    image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]
    image.save(prompt_names[idx] + '.png')