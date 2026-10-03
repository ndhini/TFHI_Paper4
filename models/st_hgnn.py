import torch
import torch.nn as nn

from models.encoder import SignalEncoder
from models.dynamic_hypergraph import DynamicHypergraph
from models.adaptive_edge import AdaptiveEdgeWeight
from models.adaptive_feature_fusion import AdaptiveFeatureFusion
from models.hgnn import HGNNLayer
from models.transformer import FeatureTransformer

class STHGNN(nn.Module):

    def __init__(
        self,
        feature_dim=256,
        num_classes=11
    ):

        super(STHGNN, self).__init__()

        # --------------------------------------------------
        # Signal Encoder
        # --------------------------------------------------
        self.encoder = SignalEncoder()
        # --------------------------------------------------
        # Transformer Feature Learning
        # --------------------------------------------------
        self.transformer = FeatureTransformer(
            token_dim=32,
            num_tokens=8,
            num_heads=4,
            num_layers=2
        )
        # --------------------------------------------------
        # Dynamic Hypergraph Construction
        # --------------------------------------------------
        self.hypergraph = DynamicHypergraph(
            max_neighbors=8,
            similarity_threshold=0.70
        )

        # --------------------------------------------------
        # Adaptive Edge Weight Learning
        # --------------------------------------------------
        self.edge_weight = AdaptiveEdgeWeight(feature_dim)

        # --------------------------------------------------
        # Hypergraph Neural Network
        # --------------------------------------------------
        self.hgnn = HGNNLayer(
            in_features=feature_dim,
            out_features=feature_dim
        )

        # --------------------------------------------------
        # Adaptive Feature Fusion
        # --------------------------------------------------
        self.feature_fusion = AdaptiveFeatureFusion(feature_dim)

        # --------------------------------------------------
        # Classifier
        # --------------------------------------------------
        self.classifier = nn.Sequential(

            nn.Linear(feature_dim, 128),

            nn.BatchNorm1d(128),

            nn.GELU(),

            nn.Dropout(0.30),

            nn.Linear(128, 64),

            nn.GELU(),

            nn.Dropout(0.20),

            nn.Linear(64, num_classes)

        )

    def forward(self, x):
       
        # --------------------------------------------------
        # Signal Encoder
        # --------------------------------------------------
        encoder_features = self.encoder(x)

        # --------------------------------------------------
        # Convert 256-D feature to 8 tokens × 32 dimensions
        # --------------------------------------------------
        tokens = encoder_features.view(
            encoder_features.size(0),
            8,
            32
        )

        # --------------------------------------------------
        # Transformer
        # --------------------------------------------------
        encoder_features = self.transformer(tokens)
        # --------------------------------------------------
        # Dynamic Hypergraph
        # --------------------------------------------------
        H = self.hypergraph(encoder_features)

        # --------------------------------------------------
        # Adaptive Edge Weight Learning
        # --------------------------------------------------
        H = self.edge_weight(encoder_features, H)

        # --------------------------------------------------
        # Hypergraph Neural Network
        # --------------------------------------------------
        hgnn_features = self.hgnn(
            encoder_features,
            H
        )

        # --------------------------------------------------
        # Adaptive Feature Fusion
        # --------------------------------------------------
        fused_features = self.feature_fusion(
            encoder_features,
            hgnn_features
        )

        # --------------------------------------------------
        # Classification
        # --------------------------------------------------
        output = self.classifier(
            fused_features
        )

        return output