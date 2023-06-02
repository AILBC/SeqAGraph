import numpy as np
from rdkit import Chem

ATOM_LIST = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe',
             'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti',
             'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb',
             'W', 'Ru', 'Nb', 'Re', 'Te', 'Rh', 'Ta', 'Tc', 'Ba', 'Bi', 'Hf', 'Mo', 'U', 'Sm', 'Os', 'Ir',
             'Ce', 'Gd', 'Ga', 'Cs', '<unk>']
ATOM_DICT = {atom: i for i, atom in enumerate(ATOM_LIST)}

ELEC_LAYER = [0, 2, 8, 8, 18, 18, 32, 32]
ELEC_LAYER_SUM = np.cumsum(ELEC_LAYER).tolist()
ATOM_GROUP_NUM = 8 + 2 + 1
ATOM_PERIOD_NUM = 7

MAX_DEGREE = 9
DEGREE = list(range(MAX_DEGREE + 1))
HYBRIDIZATION = [Chem.rdchem.HybridizationType.SP,
                 Chem.rdchem.HybridizationType.SP2,
                 Chem.rdchem.HybridizationType.SP3,
                 Chem.rdchem.HybridizationType.SP3D,
                 Chem.rdchem.HybridizationType.SP3D2]
HYBRIDIZATION_DICT = {hb: i for i, hb in enumerate(HYBRIDIZATION)}

FORMAL_CHARGE = [-1, -2, 1, 2, 0]
FC_DICT = {fc: i for i, fc in enumerate(FORMAL_CHARGE)}

VALENCE = [0, 1, 2, 3, 4, 5, 6]
VALENCE_DICT = {vl: i for i, vl in enumerate(VALENCE)}

NUM_HS = [0, 1, 3, 4, 5]
NUM_HS_DICT = {nH: i for i, nH in enumerate(NUM_HS)}

CHIRAL_TAG = [Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
              Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
              Chem.rdchem.ChiralType.CHI_UNSPECIFIED]
CHIRAL_TAG_DICT = {ct: i for i, ct in enumerate(CHIRAL_TAG)}

RS_TAG = ["R", "S", "None"]
RS_TAG_DICT = {rs: i for i, rs in enumerate(RS_TAG)}

BOND_TYPES = [None,
              Chem.rdchem.BondType.SINGLE,
              Chem.rdchem.BondType.DOUBLE,
              Chem.rdchem.BondType.TRIPLE,
              Chem.rdchem.BondType.AROMATIC]
BOND_FLOATS = [0.0, 1.0, 2.0, 3.0, 1.5]
BOND_FLOATS_DICT = {f: i for i, f in enumerate(BOND_FLOATS)}
BOND_FLOAT_TO_TYPE = {
    0.0: BOND_TYPES[0],
    1.0: BOND_TYPES[1],
    2.0: BOND_TYPES[2],
    3.0: BOND_TYPES[3],
    1.5: BOND_TYPES[4],
}

BOND_STEREO = [Chem.rdchem.BondStereo.STEREONONE,
               Chem.rdchem.BondStereo.STEREOE,
               Chem.rdchem.BondStereo.STEREOZ,]
BOND_STEREO_DICT = {st: i for i, st in enumerate(BOND_STEREO)}

AROMATIC_SIZE = 2
PERM_SIZE = 10
SUF_SIZE = 2
DUAL_TASK_SIZE = 2
REACTION_CLASS = 10

NODE_FEAT_NUM = 13
NODE_FEAT_LIST = [len(ATOM_LIST), len(DEGREE), len(FORMAL_CHARGE), len(VALENCE), len(NUM_HS),
                  len(CHIRAL_TAG), len(RS_TAG), len(HYBRIDIZATION), AROMATIC_SIZE, PERM_SIZE, SUF_SIZE, DUAL_TASK_SIZE, REACTION_CLASS]
# NODE_FEAT_NUM = 11
# NODE_FEAT_LIST = [len(ATOM_LIST), len(DEGREE), len(FORMAL_CHARGE), len(VALENCE), len(NUM_HS),
#                   len(CHIRAL_TAG), len(RS_TAG), len(HYBRIDIZATION), AROMATIC_SIZE, DUAL_TASK_SIZE, REACTION_CLASS]

NODE_FDIM = sum(NODE_FEAT_LIST)
BOND_FEAT_NUM = 4
BOND_FEAT_LIST = [len(BOND_FLOATS) - 1, len(BOND_STEREO), 2, 2]
BOND_FDIM = 1 + 4 * 3 * 2 * 2 # None + bond_type + stereo + conjugate + ring
BOND_TYPE_NUM = 5

def get_group_and_period(
    atom
):
    atomic_num = atom.GetAtomicNum()
    atom_group, atom_period = 0, 0
    for i in range(1, len(ELEC_LAYER_SUM)):
        p_min, p_max = ELEC_LAYER_SUM[i-1] + 1, ELEC_LAYER_SUM[i]
        if (atomic_num >= p_min) & (atomic_num <= p_max):
            atom_period = i
            if atomic_num - p_min <= 1: atom_group = atomic_num - p_min + 1
            elif p_max - atomic_num <= 5: atom_group = atomic_num + 8 - p_max
            elif (atomic_num >= 57) & (atomic_num <= 70): atom_group = 9
            elif (atomic_num >= 89) & (atomic_num <= 102): atom_group = 10
            else: atom_group = 11
    return [atom_group - 1, atom_period - 1]

def get_atom_feat(
    atom,
    dual_task=-1,
    reaction_class=-1,
    perm_idx=-1,
    suf_idx=-1
) -> list:
    feat = []
    symbol = atom.GetSymbol()
    feat.append(ATOM_DICT.get(symbol, ATOM_DICT['<unk>']))
    if symbol in ['<unk>'] and (atom.GetAtomicNum() <= 0 or atom.GetAtomicNum() > 118):
        feat.extend([-1] * (NODE_FEAT_NUM - 1))
    else:
        # feat.extend(get_group_and_period(atom))
        feat.append(atom.GetDegree() if atom.GetDegree() in DEGREE else MAX_DEGREE)
        feat.append(FC_DICT.get(atom.GetFormalCharge(), 4))
        feat.append(VALENCE_DICT.get(atom.GetTotalValence(), 6))
        feat.append(NUM_HS_DICT.get(atom.GetTotalNumHs(), 4))
        feat.append(CHIRAL_TAG_DICT.get(atom.GetChiralTag(), 2))
        rs_tag = atom.GetPropsAsDict().get('_CIPCode', 'None')
        feat.append(RS_TAG_DICT.get(rs_tag, 2))
        feat.append(HYBRIDIZATION_DICT.get(atom.GetHybridization(), 4))
        feat.append(int(atom.GetIsAromatic()))
        feat.append(perm_idx)
        feat.append(suf_idx)
        feat.append(dual_task)
        feat.append(reaction_class)
    return feat

def get_bond_feat(
    bond
) -> int:
    feat = []
    bond_type_list = [1.0, 2.0, 3.0, 1.5]
    bond_type = bond.GetBondTypeAsDouble()
    bond_type_id = 0
    if bond_type in bond_type_list:    
        for id, type in enumerate(bond_type_list):
            if type == bond_type:
                feat.append(id)
                bond_type_id = id + 1

        bond_ez = bond.GetStereo()
        for id, stereo in enumerate(BOND_STEREO):
            if stereo == bond_ez: feat.append(id)
        
        bond_conjugate = int(bond.GetIsConjugated())
        feat.append(bond_conjugate)

        bond_ring = int(bond.IsInRing())
        feat.append(bond_ring)
    else:
        feat.extend([-1] * BOND_FEAT_NUM)
    return feat, bond_type_id