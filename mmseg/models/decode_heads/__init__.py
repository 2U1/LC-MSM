# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0
# Modifications: Add additional decode_heads

from .aspp_head import ASPPHead
from .da_head import DAHead
from .daformer_head import DAFormerHead
from .dlv2_head import DLV2Head
from .fcn_head import FCNHead
from .isa_head import ISAHead
from .psp_head import PSPHead
from .segformer_head import SegFormerHead
from .sep_aspp_head import DepthwiseSeparableASPPHead
from .uper_head import UPerHead
from .heads import IdentityHead
from .clip_head import CLIPHead
from .fpn_head import FPNHead
from .clip_head_oa import CLIPHeadOA
from .lseg_head import LsegHead
from .lseg_head_context import LsegHeadContext
from .attention_head_context import AttentionHeadContext
from .dlv2_head_clip import DLV2HeadCLIP
from .bert_head import BertHead
from .lseg_head_context_oc import LsegHeadContextOC
from .vlc_head_context_large import VLCHeadConetextLarge
from .vlc_head_large import VLCHeadLarge
from .tigda_vlc_head import TigDAVLCHeadLarge

__all__ = [
    'FCNHead',
    'PSPHead',
    'ASPPHead',
    'UPerHead',
    'DepthwiseSeparableASPPHead',
    'DAHead',
    'DLV2Head',
    'SegFormerHead',
    'DAFormerHead',
    'ISAHead',
    'IdentityHead',
    'CLIPHead',
    'FPNHead',
    'CLIPHeadOA',
    'LsegHead',
    'LsegHeadContext',
    'AttentionHeadContext',
    'DLV2HeadCLIP',
    'BertHead',
    'LsegHeadContextOC',
    'VLCHeadConetextLarge',
    'VLCHeadLarge',
    'TigDAVLCHeadLarge',
]
