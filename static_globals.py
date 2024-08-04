BLIP_DIFFUSION="blip_diffusion"
ELITE="elite"
RIVAL="rival"
IP_ADAPTER="adapter"
FACE_IP_ADAPTER="face_adapter"
CHOSEN="chosen"
INSTANT="instant"
CHOSEN_K="chosen_k"
CHOSEN_STYLE="chosen_style"
CHOSEN_K_STYLE="chosen_k_style"
DDPO="ddpo"
DPOK="dpok"
FACE_REWARD="face_reward"
FASHION_REWARD="fashion_reward"
DDPO_MULTI="ddpo_multi"

PLACEHOLDER="<S>"

METHOD_LIST=[BLIP_DIFFUSION, ELITE, RIVAL,IP_ADAPTER,FACE_IP_ADAPTER,CHOSEN,INSTANT,DDPO,DPOK,DDPO_MULTI]

NEGATIVE="over-exposure, under-exposure, saturated, duplicate, out of frame, lowres, cropped, worst quality, low quality, jpeg artifacts, morbid, mutilated, out of frame, ugly, bad anatomy, bad proportions, deformed, blurry, duplicate"

TEXT_INPUT_IDS="text_input_ids"
CLIP_IMAGES='clip_images'
IMAGES="images" #in text_to_image_lora this is aka pixel_values

REWARD_NORMAL="reward_normal"
REWARD_TIME="reward_time"
REWARD_PARETO="reward_pareto"
REWARD_PARETO_TIME="reward_pareto_time"

REWARD_TYPE_LIST=[REWARD_NORMAL, REWARD_TIME, REWARD_PARETO,REWARD_PARETO_TIME]