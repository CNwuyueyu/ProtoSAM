import torch.nn as nn
import torch.nn.functional as F
import torch

from src.models.common import SELayer
from src.models.image_encoder import LayerNorm2d, MBConv, TinyViT
from src.models.mask_decoder import MaskDecoder
from src.models.prompt_encoder import PromptEncoder
from baseline.SAM.transformer import TwoWayTransformer


class PrototypeGuidedSelfPromptGenerator(nn.Module):

    def __init__(self, in_dim=256, out_dim=256):
        super().__init__()
        self.feature_enhance = MBConv(
            in_dim,
            in_dim,
            expand_ratio=2.0,
            activation=nn.GELU,
            drop_path=0.0,
        )
        self.se_module = SELayer(in_dim, reduction=4)

        # Concat(Fq, P) -> Conv1x1 -> +Fq
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(in_dim * 2, in_dim, kernel_size=1),
            LayerNorm2d(in_dim),
            nn.GELU(),
        )

        self.vessel_detector = nn.Sequential(
            nn.Conv2d(in_dim, in_dim // 2, 3, 1, 1, groups=max(1, in_dim // 2)),
            nn.GELU(),
            nn.Conv2d(in_dim // 2, 1, 1),
            nn.Sigmoid(),
        )

        self.prompt_encoder = nn.Sequential(
            nn.Conv2d(1, out_dim, 1),
            LayerNorm2d(out_dim),
            nn.GELU(),
        )

    def forward(self, query_feat, prototype=None):
        enhanced_feat = self.feature_enhance(query_feat)
        query_feat_refined = self.se_module(enhanced_feat)

        b, c, h, w = query_feat_refined.shape
        if prototype is None:
            prototype = torch.zeros(b, c, 1, 1, device=query_feat_refined.device, dtype=query_feat_refined.dtype)

        if prototype.dim() == 2:
            prototype = prototype.unsqueeze(-1).unsqueeze(-1)
        if prototype.shape[0] == 1 and b > 1:
            prototype = prototype.expand(b, -1, -1, -1)

        proto_map = prototype
        if proto_map.shape[-2:] != (h, w):
            proto_map = F.interpolate(proto_map, size=(h, w), mode='bilinear', align_corners=False)

        fused_feat = self.fusion_conv(torch.cat([query_feat_refined, proto_map], dim=1))
        fused_feat = fused_feat + query_feat_refined

        vessel_prob = self.vessel_detector(fused_feat)
        dense_prompt = self.prompt_encoder(vessel_prob)
        return dense_prompt, vessel_prob


class BaseSelfPromptGenerator(nn.Module):
    """ CFP """

    def __init__(self, in_dim=256, out_dim=256):
        super().__init__()
        self.feature_enhance = MBConv(
            in_dim,
            in_dim,
            expand_ratio=2.0,
            activation=nn.GELU,
            drop_path=0.0,
        )
        self.se_module = SELayer(in_dim, reduction=4)
        self.vessel_detector = nn.Sequential(
            nn.Conv2d(in_dim, in_dim // 2, 3, 1, 1, groups=max(1, in_dim // 2)),
            nn.GELU(),
            nn.Conv2d(in_dim // 2, 1, 1),
            nn.Sigmoid(),
        )
        self.prompt_encoder = nn.Sequential(
            nn.Conv2d(1, out_dim, 1),
            LayerNorm2d(out_dim),
            nn.GELU(),
        )

    def forward(self, query_feat):
        feat = self.feature_enhance(query_feat)
        feat = self.se_module(feat)
        vessel_prob = self.vessel_detector(feat)
        dense_prompt = self.prompt_encoder(vessel_prob)
        return dense_prompt, vessel_prob

class ProtoFDA_SAM(nn.Module):
    def __init__(self):
        super(ProtoFDA_SAM, self).__init__()
        img_size = 1024
        self.MedicalTinyViT = TinyViT()
        self.img_size = img_size

        self.base_self_prompt_gen = BaseSelfPromptGenerator(in_dim=256, out_dim=256)
        self.prototype_self_prompt_gen = PrototypeGuidedSelfPromptGenerator(in_dim=256, out_dim=256)

        self.prompt_encoder = PromptEncoder(
            embed_dim=256,
            image_embedding_size=(img_size // 16, img_size // 16),
            input_image_size=(img_size, img_size),
            mask_in_chans=16,
        )
        
        self.mask_decoder = MaskDecoder(
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=256,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=256,
            num_multimask_outputs=3
        )

    @staticmethod
    def _masked_average_pooling(feat, mask, eps=1e-6):
        """feat: (N,C,H,W), mask: (N,1,H,W)"""
        weighted_feat = feat * mask
        num = weighted_feat.flatten(2).sum(dim=2)
        den = mask.flatten(2).sum(dim=2).clamp_min(eps)
        return num / den

    def _build_prototype(self, support_images, support_masks, query_batch_size):
        """Build prototype with support set.

        Supported inputs:
        - support_images: (B,K,C,H,W), support_masks: (B,K,1,H,W)
        - support_images: (N,C,H,W), support_masks: (N,1,H,W)
        """
        if support_images is None or support_masks is None:
            return None

        if support_images.dim() == 5:
            b, k, c, h, w = support_images.shape
            s_img = support_images.view(b * k, c, h, w)
            s_msk = support_masks.view(b * k, 1, h, w)

            s_feat = self.MedicalTinyViT.forward_features(s_img)['neck']
            m64 = F.interpolate(s_msk.float(), size=s_feat.shape[-2:], mode='nearest')

            proto_each = self._masked_average_pooling(s_feat, m64).view(b, k, -1)
            proto = proto_each.mean(dim=1).unsqueeze(-1).unsqueeze(-1)
            return proto

        if support_images.dim() == 4:
            s_feat = self.MedicalTinyViT.forward_features(support_images)['neck']
            m64 = F.interpolate(support_masks.float(), size=s_feat.shape[-2:], mode='nearest')

            proto_each = self._masked_average_pooling(s_feat, m64)
            proto = proto_each.mean(dim=0, keepdim=True).unsqueeze(-1).unsqueeze(-1)
            if query_batch_size > 1:
                proto = proto.expand(query_batch_size, -1, -1, -1)
            return proto

        raise ValueError('Unsupported support_images shape, expected 4D or 5D tensor.')

    def forward(self, x, support_images=None, support_masks=None, training_stage: str = "auto"):

        b = x.shape[0]

        # Step 1:
        features = self.MedicalTinyViT.forward_features(x)
        
        # patch_embed |  torch.Size([8, 64, 256, 256]) | 256x256 |  64
        # layer0     | torch.Size([8, 16384, 128]) |  128x128 |  128 
        # layer1     |  torch.Size([8, 4096, 160]) |  64x64 | 160 
        # layer2     |  torch.Size([8, 4096, 320]) |  64x64 |  320 
        # layer3     |  torch.Size([8, 4096, 320]) | 64x64 |  320 
        # neck       |  torch.Size([8, 256, 64, 64]) |  64x64 |  256

        layer0_feat = features['layer0']  # (B, 16384, 128)
        layer1_feat = features['layer1']  # (B, 4096, 160)
        layer2_feat = features['layer2']  # (B, 4096, 320)
        neck_feat = features['neck']      # (B, 256, 64, 64)
        
        # Step 2
        stage = training_stage.lower()
        if stage not in ["auto", "cfp", "octa"]:
            raise ValueError("training_stage must be one of ['auto', 'cfp', 'octa']")

        use_proto = (stage == "octa") or (stage == "auto" and support_images is not None and support_masks is not None)

        if use_proto:
            prototype = self._build_prototype(support_images, support_masks, query_batch_size=b)
            dense_prompts, vessel_prob = self.prototype_self_prompt_gen(neck_feat, prototype)
        else:
            prototype = None
            dense_prompts, vessel_prob = self.base_self_prompt_gen(neck_feat)

        # Step 3
        enhanced_image_embeddings = neck_feat + dense_prompts

        prompt_masks = F.interpolate(
            vessel_prob,
            size=(self.img_size // 4, self.img_size // 4),
            mode='bilinear',
            align_corners=False,
        )
        
        # Step 4
        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=None,
            boxes=None, 
            masks=prompt_masks,
        )
        
        # Step 5

        image_pe = self.prompt_encoder.get_dense_pe()
        low_res_masks_list = []
        iou_predictions_list = []
        for i in range(b):
            low_res_i, iou_i = self.mask_decoder(
                image_embeddings=enhanced_image_embeddings[i : i + 1],
                image_pe=image_pe,
                sparse_prompt_embeddings=sparse_embeddings[i : i + 1],
                dense_prompt_embeddings=dense_embeddings[i : i + 1],
            )
            low_res_masks_list.append(low_res_i)
            iou_predictions_list.append(iou_i)

        low_res_masks = torch.cat(low_res_masks_list, dim=0)
        iou_predictions = torch.cat(iou_predictions_list, dim=0)

        best_idx = torch.argmax(iou_predictions, dim=1)
        B = low_res_masks.shape[0]
        H, W = low_res_masks.shape[2], low_res_masks.shape[3]

        best_idx_expanded = best_idx.view(B, 1, 1, 1).expand(-1, 1, H, W)
        low_res_masks = torch.gather(low_res_masks, 1, best_idx_expanded)

        # Step 6
        masks = F.interpolate(
            low_res_masks,
            size=(self.img_size, self.img_size),
            mode='bilinear',
            align_corners=False
        )
        
        return {
            'masks': masks,
            'iou_predictions': iou_predictions,
            'vessel_probs': vessel_prob,
            'dense_prompts': dense_prompts,
            'prototype': prototype,
            'layer0_feat': layer0_feat,
            'layer1_feat': layer1_feat,
            'layer2_feat': layer2_feat,
            'stage_used': 'octa' if use_proto else 'cfp',
        }


def test_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device : {device}")
    
    # create model instance and move to device
    model = ProtoFDA_SAM().to(device)
    
    # calculate total parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"total params: {total_params / 1e6:.2f}M")
    
    # test forward pass with dummy input
    test_input = torch.randn(8, 3, 1024, 1024).to(device)
    
    # forward pass
    model.eval()
    with torch.no_grad():
        outputs = model(test_input)  
    
    print("\noutput shapes:")
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")
        elif isinstance(value, dict):
            print(f"  {key}:")
            for sub_key, sub_value in value.items():
                print(f"    {sub_key}: {sub_value.shape}")
    
    print("test completed!")
    return outputs

if __name__ == "__main__":
    test_model()