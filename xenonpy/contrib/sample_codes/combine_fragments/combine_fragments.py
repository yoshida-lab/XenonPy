#  Copyright (c) 2019. stewu5. All rights reserved.
#  Use of this source code is governed by a BSD-style
#  license that can be found in the LICENSE file.

import pandas as pd
from rdkit import Chem
from xenonpy.inverse.iqspr import NGram

def combine_fragments(smis_base, smis_frag):
    """
    combine two SMILES strings with '*' as connection points, note that no proper treatment for '-*', '=*', or '#*' yet.

    Parameters
    ----------
    smis_base: str
        SMILES for combining.
        If no '*', assume connection point at the end.
        If more than one '*', the first will be picked if it's not the 1st character.
    smis_frag: str
        SMILES for combining.
        If no '*', assume connection point at the front.
        If more than one '*', the first will be picked.
    """

    # prepare NGram object for use of ext. SMILES
    ngram = NGram()

    # check position of '*'
    mols_base = Chem.MolFromSmiles(smis_base)
    if mols_base is None:
        raise RuntimeError('Invalid base SMILES!')
    idx_base = [i for i in range(mols_base.GetNumAtoms()) if mols_base.GetAtomWithIdx(i).GetSymbol() == '*']

    # rearrange base SMILES to avoid 1st char = '*' (assume no '**')
    if len(idx_base) == 1 and idx_base[0] == 0:
        smis_base_head = Chem.MolToSmiles(mols_base,rootedAtAtom=1)
    elif len(idx_base) == 0:
        smis_base_head = smis_base + '*'
    else:
        smis_base_head = smis_base

    # converge base to ext. SMILES and pick insertion location
    esmi_base = ngram.smi2esmi(smis_base_head)
    esmi_base = esmi_base[:-1]
    idx_base = esmi_base.index[esmi_base['esmi'] == '*'].tolist()
    if idx_base[0] == 0:
        if len(idx_base) == 1:
            # put treatment here
            raise RuntimeError('Probably -*, =*, and/or #* exist')
        else:
            idx_base = idx_base[1]
    else:
        idx_base = idx_base[0]

    # rearrange fragment to have 1st char = '*' and convert to ext. SMILES
    mols_frag = Chem.MolFromSmiles(smis_frag)
    if mols_frag is None:
        raise RuntimeError('Invalid frag SMILES!')
    idx_frag = [i for i in range(mols_frag.GetNumAtoms()) if mols_frag.GetAtomWithIdx(i).GetSymbol() == '*']
    if len(idx_frag) == 0: # if -*, =*, and/or #* exist, not counted as * right now
        esmi_frag = ngram.smi2esmi(smis_frag)
        # remove last '!'
        esmi_frag = esmi_frag[:-1]
    else:
        esmi_frag = ngram.smi2esmi(Chem.MolToSmiles(mols_frag,rootedAtAtom=idx_frag[0]))
        # remove leading '*' and last '!'
        esmi_frag = esmi_frag[1:-1]

    # check open rings of base SMILES
    nRing_base = esmi_base['n_ring'].loc[idx_base]

    # re-number rings in fragment SMILES
    esmi_frag['n_ring'] = esmi_frag['n_ring'] + nRing_base

    # delete '*' at the insertion location
    esmi_base = esmi_base.drop(idx_base).reset_index(drop=True)

    # combine base with the fragment
    ext_smi = pd.concat([esmi_base.iloc[:idx_base], esmi_frag, esmi_base.iloc[idx_base:]]).reset_index(drop=True)
    new_pd_row = {'esmi': '!', 'n_br': 0, 'n_ring': 0, 'substr': ['!']}
    ext_smi.append(new_pd_row, ignore_index=True)

    return ngram.esmi2smi(ext_smi)


