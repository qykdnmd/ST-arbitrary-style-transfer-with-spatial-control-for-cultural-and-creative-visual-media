"""Canonical configuration IDs and manuscript display labels."""
PROPOSED='residual_gating_spatial_alpha'
MODELS={
    PROPOSED:'Residual saliency gating + spatial α (proposed model)',
    'saliency_routed_spatial_alpha':'Saliency-routed attention + spatial α',
    'no_structure_guidance':'No structure guidance',
    'saliency_routed_scalar_alpha':'Saliency-routed attention + scalar α',
    'residual_gating_scalar_alpha':'Residual saliency gating + scalar α',
}
BASELINES={'ccstytr':'CC-StyTr (Main)','stytr2':'StyTr²','adain':'AdaIN','sanet':'SANet','cast':'CAST','aesfa':'AesFA'}
