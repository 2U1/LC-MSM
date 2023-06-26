# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

from mmseg.models.uda.dacs import DACS
from mmseg.models.uda.attention_dacs import AttentionDACS
from mmseg.models.uda.dacs_pseudo_only import DACSPseudoMix
from mmseg.models.uda.dacs_consistency import DACSConsistency
from mmseg.models.uda.dacs_cnn import DACSCNN
from mmseg.models.uda.dacs_tigda import DACSTigDA

__all__ = ['DACS', 'AttentionDACS', 'DACSPseudoMix',
           'DACSConsistency', 'AttentionDACS', 'DACSCNN', 'DACSTigDA']
