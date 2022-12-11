from diffusers import StableDiffusionPipeline
import torch

model_id = "/home/arif/Documents/design/sandpit/trainedsd/trainedallclasses/output"
model_id="/home/arif/Documents/design/sandpit/trainedsd/trainedallclasses/output/checkpoint-8002"
outputdir = '/home/arif/Documents/design/sandpit/trainedsd/trainedallclasses/images'
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")

prompts = ["A cartoon image of a wunzag bear","A cartoon image of extofer","A cartoon image of a gwala owl","A cartoon image of rawdib rabbit","A image of renwa","A cartoon image of a riglel piglet","A cartoon image of an ungera kangaroo","A cartoon image of a wunzag bear"]
prompt_names=['wunzag','extofer','gwala','rawdib','renwa','riglel','ungera','wunzag']

prompts=["a hyperrealistic painting of crashing waves sprays of water particles splashing on ricks,intense, epic, wow, panoramic, cinematic, moody, exciting, 3d stop motion, highly detailed octane render, soft lighting, celestial, professional, 35mm zeiss, IMAX trending on artstation"]
prompt_names=['basetest']
for imgnum in range(3):
    for idx,prompt in enumerate(prompts):
        image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]
        image.save(outputdir + '/' + prompt_names[idx] + str(imgnum) +  '.png')