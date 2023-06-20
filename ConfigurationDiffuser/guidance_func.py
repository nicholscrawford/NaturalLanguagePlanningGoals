import torch
import clip
from CLIPEmbedder.clip_embedder import CLIPEmbedder
from einops import rearrange
from StructDiffusion.rotation_continuity import compute_rotation_matrix_from_ortho6d

class guidance_functions:
        def __init__(self, config, device) -> None:
                self.device = device
                torch.set_default_dtype(torch.float)
                self.model, preprocess = clip.load(config.clip_model, device=device)
                torch.set_default_dtype(torch.double)
                self.embedder_model = CLIPEmbedder.load_from_checkpoint(config.embedder_checkpoint_path, clip_model=self.model).to(self.device)
                text = clip.tokenize(config.labels).to(self.device)
                self.text_features = self.model.encode_text(text)
                self.text_features = self.text_features / self.text_features.norm(dim=1, keepdim=True)

        def clip_guidance_function(self, xyz, rgbs, tfs):
                logit_scale = self.model.logit_scale.exp()
                
                points = torch.cat((xyz, rgbs), dim=3)
                xyzs = tfs[:,:,:3]
                ortho6d = tfs[:,:,3:]
                B, N, _ = ortho6d.shape
                ortho6d = rearrange(ortho6d, "B N C -> (B N) C")
                rmats = compute_rotation_matrix_from_ortho6d(ortho6d)
                rmats = rearrange(rmats, "(B N) H W -> B N H W", B=B, N=N)

                # Expand dimensions to match the homogeneous matrix shape
                xyzs = xyzs.unsqueeze(-1)
                identity = torch.eye(4).unsqueeze(0).unsqueeze(0)
                identity = identity.to(rmats.device)

                # Construct the homogeneous matrix
                tfs = torch.cat((torch.cat((rmats, xyzs), dim=-1), identity.repeat(16, 6, 1, 1)[:,:,3,:].unsqueeze(2)), dim=-2)


                my_image_features = self.embedder_model((points, tfs))
                my_image_features = my_image_features / my_image_features.norm(dim=1, keepdim=True)

                cosine_similarity = my_image_features.to(torch.half) @ self.text_features.t()
                inv_cosine_similarity = 1 - cosine_similarity
                inv_logits_per_image = logit_scale * inv_cosine_similarity 
                
                inv_logits_per_image = inv_logits_per_image.squeeze()
                # for image_similarity in inv_logits_per_image: #TODO: Figure out the best way to do this, does multiple backwards with retain graph cause artifacts?
                #         image_similarity.backward()
                self.text_features.detach_()
                #print(inv_logits_per_image.mean().item())
                return inv_logits_per_image
                
        def to_one_point_guidance_function(self, xyzs, rgbs, tfs):
                loss = torch.nn.MSELoss()
                out = loss(tfs[:, :, :3], torch.ones_like(tfs[:, :, :3]) * torch.tensor([-2, -2, 1], device=tfs.device))
                out.backward()
                return out
        
        def away_from_each_other_guidance_function(self, xyzs, rgbs, tfs):
                # Calculate pairwise distances between objects
                distances = torch.cdist(tfs[:, :, :3], tfs[:, :, :3], p=2)  # Euclidean distance
                mean_distance = torch.mean(distances)
                inverted_distance = 1 / mean_distance
                inverted_distance.backward()
                return inverted_distance