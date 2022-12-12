from diffusers import StableDiffusionPipeline
import torch


prpt0="an extremely detailed realistic cinematic photograhic image of an elven forest, cinematic masterpiece, dramatic lighting"
prpt1="an extremely detailed realistic cinematic photograhic image of jennifer lopez, cinematic masterpiece, dramatic lighting"
prpt2="an extremely detailed realistic cinematic photograhic image of a hadacod forest, cinematic masterpiece, dramatic lighting"
prp3="an extremely detailed realistic cinematic cartoon image of a hadacod forest, cinematic masterpiece, dramatic lighting"
prpt4 ="an extremely detailed realistic cinematic photograhic image of a hadacod forest in the style of an elven forest, cinematic masterpiece, dramatic lighting"
prp5="an extremely detailed realistic cinematic cartoon image of a wunzag bear, cinematic masterpiece, dramatic lighting"
prpt6="an extremely detailed realistic cinematic image of a wunzag bear, cinematic masterpiece, dramatic lighting"
prpt7="an extremely detailed realistic cinematic image of a wunzag bear in hadacod forest, cinematic masterpiece, dramatic lighting"
prp8="an extremely detailed realistic cinematic photograhic image of jennifer lopez in hadacod forest, cinematic masterpiece, dramatic lighting"

sdv2="stabilityai/stable-diffusion-2"
hc1200="/home/arif/Documents/design/sandpit/trainedsd/trainedhadacod/output/checkpoint-1200"
hc2400="/home/arif/Documents/design/sandpit/trainedsd/trainedhadacod/output/checkpoint-2400"
hc3000="/home/arif/Documents/design/sandpit/trainedsd/trainedhadacod/output"
hcwz1200="/home/arif/Documents/design/sandpit/trainedsd/trainedhadacod/hadacod2600wunzag/hadacod2400wunzag/checkpoint-1200"
hcwz2400="/home/arif/Documents/design/sandpit/trainedsd/trainedhadacod/hadacod2600wunzag/hadacod2400wunzag/checkpoint-2400"
hcwz3000="/home/arif/Documents/design/sandpit/trainedsd/trainedhadacod/hadacod2600wunzag/hadacod2400wunzag"
model_ids = [sdv2,hc1200,hc2400,hc3000,hcwz1200,hcwz2400,hcwz3000]
model_names=['sdv2','hc1200','hc2400','hc3000','hcwz1200','hcwz2400','hcwz3000']

outputdir = '/home/arif/Documents/design/sandpit/imgtests'

posprompts = [prpt0,prpt1,prpt2,prp3,prpt4,prp5,prpt6,prpt7,prp8]

prompt_names=['eleven','jlo','hadacod','hadacodcartoon','hadacodelven','wunzagcartoon','wunzag','wunzaghadacod','jlohadacod']

#posprompts=["an extremely detailed realistic cinematic photograhic image of an elven forest, cinematic masterpiece, dramatic lighting"]
#negprompts=["green grass"]
#prompt_names=['elfforest100']
for imgnum in range(3):
    for modelidx, model in enumerate(model_ids):
        pipe = StableDiffusionPipeline.from_pretrained(model, torch_dtype=torch.float16).to("cuda")
        for idx,prompt in enumerate(posprompts):
            prompt=posprompts[idx] #+ " | " + negprompts[idx] + ":-1.0"            
            imgname = prompt_names[idx]+ model_names[modelidx] + "_" + str(imgnum) +  '.png'
            image = pipe(prompt, num_inference_steps=100, guidance_scale=7.5).images[0]
            image.save(outputdir + '/' + imgname)