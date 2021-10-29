# flake8: noqa
# There's no way to ignore "F401 '...' imported but unused" warnings in this
# module, but to preserve other warnings. So, don't check this module at all.

# Copyright 2020 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import TYPE_CHECKING

from ...file_utils import (
    _LazyModule,
    is_flax_available,
    is_sentencepiece_available,
    is_tf_available,
    is_tokenizers_available,
    is_torch_available,
)


_import_structure = {
    "configuration_marian": ["MARIAN_PRETRAINED_CONFIG_ARCHIVE_MAP", "MarianConfig"],
}

if is_sentencepiece_available():
    _import_structure["tokenization_marian"] = ["MarianTokenizer"]

if is_torch_available():
    _import_structure["modeling_marian"] = [
        "MARIAN_PRETRAINED_MODEL_ARCHIVE_LIST",
        "MarianForCausalLM",
        "MarianModel",
        "MarianMTModel",
        "MarianPreTrainedModel",
        "MarianForSequenceClassification"
    ]

if is_tf_available():
    _import_structure["modeling_tf_marian"] = ["TFMarianModel", "TFMarianMTModel", "TFMarianPreTrainedModel"]

if is_flax_available():
    _import_structure["modeling_flax_marian"] = ["FlaxMarianModel", "FlaxMarianMTModel", "FlaxMarianPreTrainedModel"]
if TYPE_CHECKING:
    from .configuration_marian import MARIAN_PRETRAINED_CONFIG_ARCHIVE_MAP, MarianConfig

    if is_sentencepiece_available():
        from .tokenization_marian import MarianTokenizer

    if is_torch_available():
        from .modeling_marian import (
            MARIAN_PRETRAINED_MODEL_ARCHIVE_LIST,
            MarianForCausalLM,
            MarianModel,
            MarianMTModel,
            MarianPreTrainedModel,
            MarianForSequenceClassification
        )

    if is_tf_available():
        from .modeling_tf_marian import TFMarianModel, TFMarianMTModel, TFMarianPreTrainedModel

    if is_flax_available():
        from .modeling_flax_marian import FlaxMarianModel, FlaxMarianMTModel, FlaxMarianPreTrainedModel

else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure)
