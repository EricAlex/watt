import torch
import torch.nn as nn
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
from einops import rearrange, repeat

# --- New Module: BEV Projection Head ---
class BevProjectionHead(nn.Module):
    """
    Projects the latent world model tokens onto a BEV grid for segmentation.
    It uses a decoder-like structure where each BEV grid cell is a query.
    """
    def __init__(self, d_model, bev_height, bev_width, num_bev_classes, 
                 nhead=8, num_decoder_layers=2, dim_feedforward=1024):
        super().__init__()
        self.d_model = d_model
        self.bev_height = bev_height
        self.bev_width = bev_width
        self.num_bev_classes = num_bev_classes

        # 1. Create learnable queries for each cell in the BEV grid
        num_bev_queries = bev_height * bev_width
        self.bev_queries = nn.Parameter(torch.randn(1, num_bev_queries, d_model))
        
        # 2. Use a standard Transformer Decoder to process these queries against the world model
        decoder_layer = nn.TransformerDecoderLayer(
            d_model, nhead, dim_feedforward, activation='gelu', batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)

        # 3. An MLP to classify each BEV grid cell from the decoder's output
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, num_bev_classes)
        )

    def forward(self, latent_world_model):
        """
        Args:
            latent_world_model (torch.Tensor): Output from the main encoder.
                                                Shape: (B, NumTokens, D_model)

        Returns:
            torch.Tensor: BEV segmentation logits.
                          Shape: (B, NumBevClasses, BevHeight, BevWidth)
        """
        b = latent_world_model.shape[0]
        
        # Repeat the BEV queries for each item in the batch
        # Shape: (B, BevHeight*BevWidth, D_model)
        bev_queries = repeat(self.bev_queries, '1 n d -> b n d', b=b)

        # Pass queries and the latent world model to the decoder.
        # This allows each BEV grid point to "look at" the entire scene context.
        # Shape: (B, BevHeight*BevWidth, D_model)
        bev_features = self.decoder(tgt=bev_queries, memory=latent_world_model)
        
        # Classify each point on the grid
        # Shape: (B, BevHeight*BevWidth, NumBevClasses)
        bev_logits = self.classifier(bev_features)

        # Reshape to the final grid format for the loss function
        # Shape: (B, NumBevClasses, BevHeight, BevWidth)
        bev_logits = rearrange(bev_logits, 'b (h w) c -> b c h w', h=self.bev_height, w=self.bev_width)
        
        return bev_logits

# --- New Module: Agent Detection Head ---
class AgentDetectionHead(nn.Module):
    """
    Detects dynamic agents in the scene using a DETR-like approach.
    Each of the 'agent_queries' learns to specialize in finding one agent.
    """
    def __init__(self, d_model, max_agents=50, num_agent_classes=4,
                 nhead=8, num_decoder_layers=2, dim_feedforward=1024):
        super().__init__()
        self.max_agents = max_agents
        
        # 1. Learnable queries, one for each potential agent we can detect.
        self.agent_queries = nn.Parameter(torch.randn(1, max_agents, d_model))
        
        # 2. A Transformer decoder to process these queries against the world model.
        decoder_layer = nn.TransformerDecoderLayer(
            d_model, nhead, dim_feedforward, activation='gelu', batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)

        # --- STABILITY FIX: Add a normalization layer ---
        self.output_norm = nn.LayerNorm(d_model)

        # 3. Output heads for each query.
        #    - Classification: car, pedestrian, cyclist, or "no_object"
        self.class_head = nn.Linear(d_model, num_agent_classes)
        #    - Bounding Box: (center_x, center_y, width, height, yaw)
        self.bbox_head = nn.Linear(d_model, 5)

    def forward(self, latent_world_model):
        """
        Args:
            latent_world_model (torch.Tensor): (B, NumTokens, D_model)
        Returns:
            dict: A dictionary containing predicted class logits and bounding boxes.
        """
        b = latent_world_model.shape[0]
        agent_queries = repeat(self.agent_queries, '1 n d -> b n d', b=b)
        
        # Let agent queries attend to the latent world model
        agent_features = self.decoder(tgt=agent_queries, memory=latent_world_model)
        
        # --- Apply the normalization before the final projection heads ---
        agent_features_norm = self.output_norm(agent_features)
        
        class_logits = self.class_head(agent_features_norm)
        pred_boxes = self.bbox_head(agent_features_norm).sigmoid()
        
        return {'pred_logits': class_logits, 'pred_boxes': pred_boxes}

# A helper function to encapsulate the logic for positional embeddings.
# This could be sinusoidal, learned, or other variants. We'll use a simple learned one.
def get_positional_embeddings(sequence_length, d_model):
    """
    Returns learned positional embeddings.
    """
    return nn.Parameter(torch.randn(1, sequence_length, d_model))


class ImageEncoder(nn.Module):
    """
    Encodes a batch of multi-camera, multi-timestamp images into feature patches.
    This module acts as the perception backbone.
    """
    def __init__(self, d_model, in_channels=3, backbone_name='convnext_tiny'):
        super().__init__()
        self.d_model = d_model

        # Best Choice: Use a powerful, pre-trained backbone like ConvNeXt.
        # We grab the feature extractor part, discarding the final classification head.
        weights = ConvNeXt_Tiny_Weights.DEFAULT
        backbone = convnext_tiny(weights=weights)
        
        # We'll use the output of the 3rd stage (index 6) which has rich features.
        self.feature_extractor = nn.Sequential(*list(backbone.features.children())[:-2])
        
        # Determine the output channel dimension from the backbone
        with torch.no_grad():
            dummy_input = torch.randn(1, in_channels, 224, 224)
            dummy_output = self.feature_extractor(dummy_input)
            self.backbone_out_dim = dummy_output.shape[1]

        # A linear projection to map backbone features to the model's hidden dimension
        self.projection = nn.Linear(self.backbone_out_dim, d_model)

    def forward(self, images):
        """
        Args:
            images (torch.Tensor): Input images of shape (B, T, C, H, W, 3)
                                   B=batch, T=time, C=cameras

        Returns:
            torch.Tensor: Feature tokens of shape (B, T*C*NumPatches, D_model)
        """
        b, t, c, h, w, _ = images.shape
        
        # Permute and reshape for backbone processing: (B, T, C, H, W, 3) -> (B*T*C, 3, H, W)
        x = rearrange(images, 'b t c h w d -> (b t c) d h w')
        
        # Get feature maps from the backbone
        # Shape: (B*T*C, BackboneOutDim, H', W')
        features = self.feature_extractor(x)
        
        # Flatten spatial dimensions into patches and project
        # Shape: (B*T*C, BackboneOutDim, H'*W') -> (B*T*C, H'*W', BackboneOutDim)
        patches = rearrange(features, 'btc d h w -> btc (h w) d')
        
        # Project to the model's dimension
        # Shape: (B*T*C, H'*W', D_model)
        tokens = self.projection(patches)

        # Reshape to group by batch element for the transformer
        # Shape: (B, T*C*H'*W', D_model)
        final_tokens = rearrange(tokens, '(b t c) p d -> b (t c p) d', b=b, t=t, c=c)
        
        return final_tokens


class WorldAwareTrajectoryTransformer(nn.Module):
    def __init__(self, num_cameras=6, num_timesteps=4, num_trajectory_points=10,
                 d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6,
                 dim_feedforward=2048, dropout=0.1,
                 # New parameters for the BEV head
                 bev_height=100, bev_width=100, num_bev_classes=3,
                 # New parameters for the agent detection head
                 max_agents=50, num_agent_classes=4):
        super().__init__()
        # ... (all previous initializations remain the same) ...
        self.d_model = d_model
        self.num_cameras = num_cameras
        self.num_timesteps = num_timesteps
        self.num_trajectory_points = num_trajectory_points

        # 1. Perception and World Model Encoder
        self.image_encoder = ImageEncoder(d_model=d_model)
        self.camera_embedding = nn.Embedding(num_cameras, d_model)
        self.time_embedding = nn.Embedding(num_timesteps, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation='gelu', batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)

        # 2. Agent / Planner Decoder
        self.planning_queries = nn.Parameter(torch.randn(1, num_trajectory_points, d_model))
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation='gelu', batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)
        
        # 3. Output Heads
        self.trajectory_head = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.ReLU(), nn.Linear(d_model // 2, 2))
        
        # --- Auxiliary Heads (Now with a real implementation for BEV) ---
        self.bev_segmentation_head = BevProjectionHead(
            d_model=d_model,
            bev_height=bev_height,
            bev_width=bev_width,
            num_bev_classes=num_bev_classes,
            nhead=nhead
        )
        self.agent_detection_head = AgentDetectionHead(
            d_model=d_model,
            max_agents=max_agents,
            num_agent_classes=num_agent_classes,
            nhead=nhead
        )

    def forward(self, images):
        # ... (The entire forward pass up to the latent_world_model is identical) ...
        b, t, c, h, w, _ = images.shape
        image_tokens = self.image_encoder(images)
        num_patches_per_image = image_tokens.shape[1] // (t * c)
        time_indices = torch.arange(t, device=images.device).repeat_interleave(c * num_patches_per_image)
        cam_indices = torch.arange(c, device=images.device).repeat_interleave(num_patches_per_image).repeat(t)
        time_indices = repeat(time_indices, '... -> b ...', b=b)
        cam_indices = repeat(cam_indices, '... -> b ...', b=b)
        contextualized_tokens = image_tokens + self.time_embedding(time_indices) + self.camera_embedding(cam_indices)
        latent_world_model = self.transformer_encoder(contextualized_tokens)
        
        # --- DECODER (Planning) ---
        planning_queries = repeat(self.planning_queries, '1 n d -> b n d', b=b)
        plan_features = self.transformer_decoder(tgt=planning_queries, memory=latent_world_model)
        predicted_trajectory = self.trajectory_head(plan_features)
        
        # --- AUXILIARY OUTPUTS (Now calling the real BEV head) ---
        bev_map_logits = self.bev_segmentation_head(latent_world_model)
        agent_predictions = self.agent_detection_head(latent_world_model)
        
        return {
            "trajectory": predicted_trajectory,
            "bev_map_logits": bev_map_logits, # Renamed for clarity
            "agent_predictions": agent_predictions
        }


# --- Example Usage & Sanity Check ---
if __name__ == '__main__':
    # Define model parameters based on our design
    BATCH_SIZE = 2
    NUM_TIMESTEPS = 4
    NUM_CAMERAS = 6
    IMG_HEIGHT = 224
    IMG_WIDTH = 224
    
    # Hyperparameters
    D_MODEL = 256
    NHEAD = 8
    NUM_ENCODER_LAYERS = 4
    NUM_DECODER_LAYERS = 4
    MAX_AGENTS = 50
    NUM_AGENT_CLASSES = 4
    
    # Instantiate the model
    model = WorldAwareTrajectoryTransformer(
        num_cameras=NUM_CAMERAS,
        num_timesteps=NUM_TIMESTEPS,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        num_decoder_layers=NUM_DECODER_LAYERS,
        max_agents=MAX_AGENTS,
        num_agent_classes=NUM_AGENT_CLASSES
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    print(f"Model created and moved to {device}.")
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {num_params / 1e6:.2f}M")
    
    # Create a dummy input tensor
    dummy_input_images = torch.randn(
        BATCH_SIZE, NUM_TIMESTEPS, NUM_CAMERAS, IMG_HEIGHT, IMG_WIDTH, 3
    ).to(device)
    
    # Perform a forward pass
    print("\nPerforming a dummy forward pass...")
    with torch.no_grad():
        model.eval()
        output = model(dummy_input_images)
    
    # --- CORRECTED Verification Loop ---
    print("Forward pass successful!")
    print("\n--- Output Shapes ---")
    for key, value in output.items():
        if key == "agent_predictions":
            print(f"{key}:")
            for sub_key, sub_value in value.items():
                print(f"  - {sub_key}: {sub_value.shape}")
        elif isinstance(value, torch.Tensor):
            print(f"{key}: {value.shape}")
        else:
            # This branch should ideally not be hit now
            print(f"{key}: {value}")
            
    # --- Add Assertions for All Heads ---
    print("\n--- Verifying Output Shapes ---")
    
    # 1. Trajectory Head
    expected_traj_shape = (BATCH_SIZE, model.num_trajectory_points, 2)
    assert output['trajectory'].shape == expected_traj_shape, \
        f"Trajectory shape is incorrect! Got {output['trajectory'].shape}, expected {expected_traj_shape}"
    print("Trajectory output shape is correct.")

    # 2. BEV Head
    expected_bev_shape = (BATCH_SIZE, model.bev_segmentation_head.num_bev_classes, 
                          model.bev_segmentation_head.bev_height, model.bev_segmentation_head.bev_width)
    assert output['bev_map_logits'].shape == expected_bev_shape, \
        f"BEV logits shape is incorrect! Got {output['bev_map_logits'].shape}, expected {expected_bev_shape}"
    print("BEV logits output shape is correct.")
    
    # 3. Agent Detection Head
    agent_preds = output['agent_predictions']
    expected_logits_shape = (BATCH_SIZE, MAX_AGENTS, NUM_AGENT_CLASSES)
    expected_boxes_shape = (BATCH_SIZE, MAX_AGENTS, 5)
    
    assert agent_preds['pred_logits'].shape == expected_logits_shape, \
        f"Agent logits shape is incorrect! Got {agent_preds['pred_logits'].shape}, expected {expected_logits_shape}"
    print("Agent logits output shape is correct.")
    
    assert agent_preds['pred_boxes'].shape == expected_boxes_shape, \
        f"Agent boxes shape is incorrect! Got {agent_preds['pred_boxes'].shape}, expected {expected_boxes_shape}"
    print("Agent boxes output shape is correct.")
