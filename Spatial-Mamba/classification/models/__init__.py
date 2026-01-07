from .spatialmamba import SpatialMamba, Backbone_SpatialMamba

def build_model(config, is_pretrain=False):
    model_type = str(config.MODEL.TYPE).lower()

    # origin
    if model_type in ["spatial_mamba", "spatialmamba"]:
        model = SpatialMamba(
            in_chans=config.MODEL.SPATIALMAMBA.IN_CHANS,
            patch_size=config.MODEL.SPATIALMAMBA.PATCH_SIZE,
            num_classes=config.MODEL.NUM_CLASSES,
            depths=config.MODEL.SPATIALMAMBA.DEPTHS,
            dims=config.MODEL.SPATIALMAMBA.EMBED_DIM,
            d_state=config.MODEL.SPATIALMAMBA.D_STATE,
            dt_init=config.MODEL.SPATIALMAMBA.DT_INIT,
            mlp_ratio=config.MODEL.SPATIALMAMBA.MLP_RATIO,
            drop_rate=config.MODEL.DROP_RATE,
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
        )
        return model

    if model_type in ["backbone_spatial_mamba", "backbone_spatialmamba"]:
        return Backbone_SpatialMamba(
            out_indices=getattr(config.MODEL, "OUT_INDICES", (0,1,2,3)),
            pretrained=getattr(config.MODEL, "PRETRAINED", None),
            norm_layer=getattr(config.MODEL, "NORM_LAYER", "ln"),
            img_size=getattr(config.DATA, "IMG_SIZE", 224),
            in_chans=config.MODEL.SPATIALMAMBA.IN_CHANS,
            patch_size=config.MODEL.SPATIALMAMBA.PATCH_SIZE,
            num_classes=config.MODEL.NUM_CLASSES,
            depths=config.MODEL.SPATIALMAMBA.DEPTHS,
            dims=config.MODEL.SPATIALMAMBA.EMBED_DIM,
        )

    return None
