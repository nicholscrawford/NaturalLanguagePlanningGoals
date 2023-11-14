import torch
import clip
from PIL import Image
from CLIPEmbedder.clip_embedder import CLIPEmbedder
from Data.basic_writerdatasets_st import CLIPEmbedderDataset

"""
This is meant to evaluate trained embedders, as compared to the stock image embedder.
"""
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
dataset = CLIPEmbedderDataset(preprocess, device, ["CLIPEmbedder/Tests/Data"])
embedder_model = CLIPEmbedder.load_from_checkpoint("/home/nicholscrawfordtaylor/Experiments/NLPGoals/embedder_experiments/jul4/2023_09_05-11:57:34/lightning_logs/version_0/checkpoints/epoch=205-step=247406.ckpt", clip_model=model).to(device)
use_logits = False

def get_eval_strs(
            default = [
                "soup on the top right", 
                "jello box on the middle shelf",
                "middle shelf jello box",
                "cleaner on the top shelf",
                "blue pitcher on the top right",
                "blue pitcher, top right",
                "mustard on the top left",
                "spam on the bottom right",
                "cleaner on top", 
                "coffee can in the middle shelf",
                "scissors on the bottom shelf",
                "bowl on the middle shelf",
                "cheez its on the bottom shelf",
                "cheezits on the bottom", 
                "wood block on the bottom shelf",
                "a man walking a dog"
                ]
            ):
    print("Enter strings to evaluate followed by a 'end' or 'e' or 'd' to use the default strings.")
    eval_strs = []
    txt = input()
    while txt != "end" and txt != "e":
        if txt == "d":
            return default
        eval_strs.append(txt)
        txt = input()
    return eval_strs

def eval(eval_strs, pth):
    with torch.no_grad():
        text = clip.tokenize(eval_strs).to(device)
        (x, y) = dataset.get_from_image_name(pth)
        y = y.to(device).unsqueeze(0)
        image_features = model.encode_image(y)
        my_image_features = embedder_model((x[0].to(device).unsqueeze(0), x[1].to(device).unsqueeze(0)))
        text_features = model.encode_text(text)
        
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        my_image_features = my_image_features / my_image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        
        logit_scale = model.logit_scale.exp() if use_logits else 1

        inverse = True
        if inverse:
            cosine_similarity = image_features @ text_features.t()
            inv_cosine_similarity = 1 - cosine_similarity
            logits_per_image = logit_scale * inv_cosine_similarity
            logits_per_text = logits_per_image.t()
            
            my_cosine_similarity = my_image_features.to(torch.half) @ text_features.t()
            my_inv_cosine_similarity = 1 - my_cosine_similarity
            my_logits_per_image = logit_scale * my_inv_cosine_similarity
            my_logits_per_text = my_logits_per_image.t()

        else:
        # cosine similarity as logits
            logit_scale = model.logit_scale.exp()
            logits_per_image = logit_scale * image_features @ text_features.t()
            logits_per_text = logits_per_image.t()
            
            my_logits_per_image = logit_scale * my_image_features.to(torch.half) @ text_features.t()
            my_logits_per_text = my_logits_per_image.t()
        
        
        
        print("Text Label\t\t\t\t\tCLIP Score\tMy Embedder Score")
        print("----------------------------------------------------------------------------")
        for str, score, my_score in zip(eval_strs, logits_per_text, my_logits_per_text):
            print(f"{str:<40}\t{score.item():<10.6f}\t{my_score.item():<10.6f}")    
            
def main():
    soup_on_top_image = "rgb_67.png"
    cleaner_on_top_image = "rgb_102.png"
    cheezits_on_bottom = "rgb_117.png"
    pitcher_on_top_right = "rgb_127.png"

    eval_strs = get_eval_strs()
    
    print(f"Testing soup_on_top_image {soup_on_top_image}")
    eval(eval_strs, soup_on_top_image)
        
    print(f"Testing cleaner_on_top_image {cleaner_on_top_image}")
    eval(eval_strs, cleaner_on_top_image)
    
    print(f"Testing cheezits_on_bottom {cheezits_on_bottom}")
    eval(eval_strs, cheezits_on_bottom)
   
    print(f"Testing pitcher_on_top_right {pitcher_on_top_right}")
    eval(eval_strs, pitcher_on_top_right)

if __name__ == "__main__":
    main()