import torch
import clip
from PIL import Image
from CLIPEmbedder.clip_embedder import CLIPEmbedder
from Data.basic_writerdatasets_mem import CLIPEmbedderDataset

"""
This is meant to evaluate trained embedders, as compared to the stock image embedder.
"""
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
dataset = CLIPEmbedderDataset(preprocess, device, ["CLIPEmbedder/Tests/Data"])
embedder_model = CLIPEmbedder.load_from_checkpoint("/home/nichols/Experiments/NLPGoals/embedder_experiments/lightning_logs/version_49970/checkpoints/epoch=352-step=434896.ckpt", clip_model=model)

def get_eval_strs(
            default = [
                "a set of objects placed in a circle", 
                "a set of objects placed in a straight line",
                "a set of objects placed in a line",
                "place the objects in a circle",
                "place the objects in a straight line",
                "place the objects in a line",
                "a set of objects in a circle",
                "a set of objects in a straight line", 
                "a set of objects in a line",
                "objects in a circle",
                "objects in a straight line",
                "objects in a line",
                "objects on the top shelf", 
                "objects on the ground",
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
        my_image_features = image_features / my_image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

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
    circle_image = "rgb_0002.png"
    line_image = "rgb_0000.png"
    
    
    print(f"Testing circle image {circle_image}")
    eval_strs = get_eval_strs()
    eval(eval_strs, circle_image)
        
    print(f"Testing line image {line_image}")
    eval_strs = get_eval_strs()
    eval(eval_strs, line_image)
   
        
if __name__ == "__main__":
    main()