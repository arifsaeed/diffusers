git clone https://github.com/arifsaeed/diffusers.git
pip install git+file:///workspace/diffusers#egg=diffusers
pip install gdown
cd diffusers/examples/dreambooth
gdown 1tIKitYzkPXTvX9jzED4Yf0K-5EqHIcDL
gdown 1QCE-fIf5mMZuljLJtkR01uIWQqXu7hgt
gdown 1zG7KcWos_D22N_qMAVKGdq2JYKOne390


pip install -U -r requirements.txt


export MODEL_NAME="/workspace/diffusers/examples/dreambooth/output/checkpoint-2400"
export INSTANCE_DIR="/workspace/diffusers/examples/dreambooth/imagesbyinstance/wunzag"
export INSTACE_PROMPT_DIR="/workspace/diffusers/examples/dreambooth/prompts/wunzagwithHadacod"
export CLASS_PROMPT_DIR="/workspace/diffusers/examples/dreambooth/prompts/bear"
export CLASS_DIR="/workspace/diffusers/examples/dreambooth/imagesbyclass/bear"
export OUTPUT_DIR="/workspace/diffusers/examples/dreambooth/hadacod2400wunzag"


accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --train_text_encoder \
  --instance_data_dir=$INSTANCE_DIR \
  --class_data_dir=$CLASS_DIR \
  --output_dir=$OUTPUT_DIR \
  --with_prior_preservation --prior_loss_weight=1.0 \
  --instance_prompt=$INSTACE_PROMPT_DIR \
  --class_prompt=$CLASS_PROMPT_DIR \
  --resolution=768 \
  --train_batch_size=1 \
  --gradient_checkpointing \
  --learning_rate=2e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_class_images=180 \
  --max_train_steps=3000 \
  --save_steps=1200 \
  --modeltoken='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'
