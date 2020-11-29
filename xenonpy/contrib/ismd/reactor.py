#  Copyright (c) 2019. yoshida-lab. All rights reserved.
#  Use of this source code is governed by a BSD-style
#  license that can be found in the LICENSE file.

from __future__ import division, unicode_literals
import argparse

from onmt.translate.translator import build_translator
from onmt.translate.translator import Translator

import onmt.inputters
import onmt.translate
import onmt
import onmt.model_builder
import onmt.modules


class Reactor():

    def __init__(self, model: Translator):
        """
        A chemical reaction prediction model
        ----------
        Parameters:
            model : A molecular transformer model for reaction prediction
        """
        self._model = model

    def smi_tokenizer(self, smi) -> str:
        """
        Tokenize a SMILES molecule or reaction
        ----------
        Parameters:
            smi : SMILES
        Returns:
            tokenized SMILES
        """
        import re
        pattern = "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
        regex = re.compile(pattern)
        tokens = [token for token in regex.findall(smi)]
        assert smi == ''.join(tokens)
        return ' '.join(tokens)

    def react(self, reactant_list, batch_size=64) -> list:
        """
        Tokenize a SMILES molecule or reaction
        ----------
        Parameters:
            reactant_list : list of reactant 
        Returns:
            product_list: all_predictions is a list of `batch_size` lists of `n_best` predictions
        """
        reactant_token_list = [self.smi_tokenizer(s) for s in reactant_list]
        _, product_list = self._model.translate(src=reactant_token_list,
                                               src_dir='',
                                               batch_size=batch_size,
                                               attn_debug=False)
        product_list = [s[0].replace(' ', '') for s in product_list]

        return product_list


def load_reactor(max_length=200, *, device_id=-1, model_path='') -> Reactor:
    """
    Loader of reaction prediction model
    ----------
    Parameters:
        model : A molecular transformer model for reaction prediction
        device_id : cpu: -1; gpu: 0,1,2,3...
        model_path : path of transformer file.
    Returns:
        Reactor : A chemical reaction prediction model
    """
    parser = argparse.ArgumentParser(description='translate.py', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    #onmt.opts.add_md_help_argument(parser)
    onmt.opts.translate_opts(parser)
    opt = parser.parse_args([
        '-src', 'dummy_src', '-model', model_path, '-replace_unk', '-max_length',
        str(max_length), '-gpu',
        str(device_id)
    ])
    translator = build_translator(opt, report_score=False)

    return Reactor(translator)
