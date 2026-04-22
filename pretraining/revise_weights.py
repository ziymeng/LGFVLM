import torch

path = "./LaMed/script/LaMed/output/CLIP-0002-1e-5/model_params.bin"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_params = torch.load(path, map_location=device)
store_dict = {}
for key in model_params.keys():
    if 'vision_encoder' in key:
        store_dict[key.replace("vision_encoder.", "")] = model_params[key]
    elif 'mm_vision_proj' in key:
        store_dict[key.replace("mm_vision_proj.", "")] = model_params[key]
        
for key in store_dict.keys():
    print(key)