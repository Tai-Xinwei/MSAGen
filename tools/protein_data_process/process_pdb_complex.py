#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import gzip
import io
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
from absl import logging
from Bio.Data import PDBData
from Bio.PDB import MMCIF2Dict
from ogb.utils.features import atom_to_feature_vector
from ogb.utils.features import bond_to_feature_vector
from rdkit import Chem
from rdkit import RDLogger

from mmcif_parsing import AtomCartn
from mmcif_parsing import mmcif_loop_to_dict
from mmcif_parsing import mmcif_loop_to_list
from mmcif_parsing import MmcifObject
from mmcif_parsing import parse_structure
from mmcif_parsing import ResidueAtPosition
from residue_constants import ATOMORDER
from residue_constants import RESIDUEATOMS


STDRESIDUES = {'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL', 'UNK', 'A', 'C', 'G', 'U', 'DA', 'DC', 'DG', 'DT', 'N', 'DN'}
EXCEPTIONS = {'144', '15P', '1PE', '2F2', '2JC', '3HR', '3SY', '7N5', '7PE', '9JE', 'AAE', 'ABA', 'ACE', 'ACN', 'ACT', 'ACY', 'AZI', 'BAM', 'BCN', 'BCT', 'BDN', 'BEN', 'BME', 'BO3', 'BTB', 'BTC', 'BU1', 'C8E', 'CAD', 'CAQ', 'CBM', 'CCN', 'CIT', 'CL', 'CLR', 'CM', 'CMO', 'CO3', 'CPT', 'CXS', 'D10', 'DEP', 'DIO', 'DMS', 'DN', 'DOD', 'DOX', 'EDO', 'EEE', 'EGL', 'EOH', 'EOX', 'EPE', 'ETF', 'FCY', 'FJO', 'FLC', 'FMT', 'FW5', 'GOL', 'GSH', 'GTT', 'GYF', 'HED', 'IHP', 'IHS', 'IMD', 'IOD', 'IPA', 'IPH', 'LDA', 'MB3', 'MEG', 'MES', 'MLA', 'MLI', 'MOH', 'MPD', 'MRD', 'MSE', 'MYR', 'N', 'NA', 'NH2', 'NH4', 'NHE', 'NO3', 'O4B', 'OHE', 'OLA', 'OLC', 'OMB', 'OME', 'OXA', 'P6G', 'PE3', 'PE4', 'PEG', 'PEO', 'PEP', 'PG0', 'PG4', 'PGE', 'PGR', 'PLM', 'PO4', 'POL', 'POP', 'PVO', 'SAR', 'SCN', 'SEO', 'SEP', 'SIN', 'SO4', 'SPD', 'SPM', 'SR', 'STE', 'STO', 'STU', 'TAR', 'TBU', 'TME', 'TPO', 'TRS', 'UNK', 'UNL', 'UNX', 'UPL', 'URE'}
GLYCANS = {'045', '05L', '07E', '07Y', '08U', '09X', '0BD', '0H0', '0HX', '0LP', '0MK', '0NZ', '0UB', '0V4', '0WK', '0XY', '0YT', '10M', '12E', '145', '147', '149', '14T', '15L', '16F', '16G', '16O', '17T', '18D', '18O', '1CF', '1FT', '1GL', '1GN', '1LL', '1S3', '1S4', '1SD', '1X4', '20S', '20X', '22O', '22S', '23V', '24S', '25E', '26O', '27C', '289', '291', '293', '2DG', '2DR', '2F8', '2FG', '2FL', '2GL', '2GS', '2H5', '2HA', '2M4', '2M5', '2M8', '2OS', '2WP', '2WS', '32O', '34V', '38J', '3BU', '3DO', '3DY', '3FM', '3GR', '3HD', '3J3', '3J4', '3LJ', '3LR', '3MG', '3MK', '3R3', '3S6', '3SA', '3YW', '40J', '42D', '445', '44S', '46D', '46Z', '475', '48Z', '491', '49A', '49S', '49T', '49V', '4AM', '4CQ', '4GC', '4GL', '4GP', '4JA', '4N2', '4NN', '4QY', '4R1', '4RS', '4SG', '4UZ', '4V5', '50A', '51N', '56N', '57S', '5GF', '5GO', '5II', '5KQ', '5KS', '5KT', '5KV', '5L3', '5LS', '5LT', '5MM', '5N6', '5QP', '5SP', '5TH', '5TJ', '5TK', '5TM', '61J', '62I', '64K', '66O', '6BG', '6C2', '6DM', '6GB', '6GP', '6GR', '6K3', '6KH', '6KL', '6KS', '6KU', '6KW', '6LA', '6LS', '6LW', '6MJ', '6MN', '6PZ', '6S2', '6UD', '6YR', '6ZC', '73E', '79J', '7CV', '7D1', '7GP', '7JZ', '7K2', '7K3', '7NU', '83Y', '89Y', '8B7', '8B9', '8EX', '8GA', '8GG', '8GP', '8I4', '8LR', '8OQ', '8PK', '8S0', '8YV', '95Z', '96O', '98U', '9AM', '9C1', '9CD', '9GP', '9KJ', '9MR', '9OK', '9PG', '9QG', '9S7', '9SG', '9SJ', '9SM', '9SP', '9T1', '9T7', '9VP', '9WJ', '9WN', '9WZ', '9YW', 'A0K', 'A1Q', 'A2G', 'A5C', 'A6P', 'AAL', 'ABD', 'ABE', 'ABF', 'ABL', 'AC1', 'ACR', 'ACX', 'ADA', 'AF1', 'AFD', 'AFO', 'AFP', 'AGL', 'AH2', 'AH8', 'AHG', 'AHM', 'AHR', 'AIG', 'ALL', 'ALX', 'AMG', 'AMN', 'AMU', 'AMV', 'ANA', 'AOG', 'AQA', 'ARA', 'ARB', 'ARI', 'ARW', 'ASC', 'ASG', 'ASO', 'AXP', 'AXR', 'AY9', 'AZC', 'B0D', 'B16', 'B1H', 'B1N', 'B2G', 'B4G', 'B6D', 'B7G', 'B8D', 'B9D', 'BBK', 'BBV', 'BCD', 'BDF', 'BDG', 'BDP', 'BDR', 'BEM', 'BFN', 'BG6', 'BG8', 'BGC', 'BGL', 'BGN', 'BGP', 'BGS', 'BHG', 'BM3', 'BM7', 'BMA', 'BMX', 'BND', 'BNG', 'BNX', 'BO1', 'BOG', 'BQY', 'BS7', 'BTG', 'BTU', 'BW3', 'BWG', 'BXF', 'BXP', 'BXX', 'BXY', 'BZD', 'C3B', 'C3G', 'C3X', 'C4B', 'C4W', 'C5X', 'CBF', 'CBI', 'CBK', 'CDR', 'CE5', 'CE6', 'CE8', 'CEG', 'CEZ', 'CGF', 'CJB', 'CKB', 'CKP', 'CNP', 'CR1', 'CR6', 'CRA', 'CT3', 'CTO', 'CTR', 'CTT', 'D1M', 'D5E', 'D6G', 'DAF', 'DAG', 'DAN', 'DDA', 'DDL', 'DEG', 'DEL', 'DFR', 'DFX', 'DG0', 'DGO', 'DGS', 'DGU', 'DJB', 'DJE', 'DK4', 'DKX', 'DKZ', 'DL6', 'DLD', 'DLF', 'DLG', 'DNO', 'DO8', 'DOM', 'DPC', 'DQR', 'DR2', 'DR3', 'DR5', 'DRI', 'DSR', 'DT6', 'DVC', 'DYM', 'E3M', 'E5G', 'EAG', 'EBG', 'EBQ', 'EEN', 'EEQ', 'EGA', 'EMP', 'EMZ', 'EPG', 'EQP', 'EQV', 'ERE', 'ERI', 'ETT', 'EUS', 'F1P', 'F1X', 'F55', 'F58', 'F6P', 'F8X', 'FBP', 'FCA', 'FCB', 'FCT', 'FDP', 'FDQ', 'FFC', 'FFX', 'FIF', 'FK9', 'FKD', 'FMF', 'FMO', 'FNG', 'FNY', 'FRU', 'FSA', 'FSI', 'FSM', 'FSW', 'FUB', 'FUC', 'FUD', 'FUF', 'FUL', 'FUY', 'FVQ', 'FX1', 'FYJ', 'G0S', 'G16', 'G1P', 'G20', 'G28', 'G2F', 'G3F', 'G3I', 'G4D', 'G4S', 'G6D', 'G6P', 'G6S', 'G7P', 'G8Z', 'GAA', 'GAC', 'GAD', 'GAF', 'GAL', 'GAT', 'GBH', 'GC1', 'GC4', 'GC9', 'GCB', 'GCD', 'GCN', 'GCO', 'GCS', 'GCT', 'GCU', 'GCV', 'GCW', 'GDA', 'GDL', 'GE1', 'GE3', 'GFP', 'GIV', 'GL0', 'GL1', 'GL2', 'GL4', 'GL5', 'GL6', 'GL7', 'GL9', 'GLA', 'GLC', 'GLD', 'GLF', 'GLG', 'GLO', 'GLP', 'GLS', 'GLT', 'GM0', 'GMB', 'GMH', 'GMT', 'GMZ', 'GN1', 'GN4', 'GNS', 'GNX', 'GP0', 'GP1', 'GP4', 'GPH', 'GPK', 'GPM', 'GPO', 'GPQ', 'GPU', 'GPV', 'GPW', 'GQ1', 'GRF', 'GRX', 'GS1', 'GS9', 'GTK', 'GTM', 'GTR', 'GU0', 'GU1', 'GU2', 'GU3', 'GU4', 'GU5', 'GU6', 'GU8', 'GU9', 'GUF', 'GUL', 'GUP', 'GUZ', 'GXL', 'GXV', 'GYE', 'GYG', 'GYP', 'GYU', 'GYV', 'GZL', 'H1M', 'H1S', 'H2P', 'H3S', 'H53', 'H6Q', 'H6Z', 'HBZ', 'HD4', 'HNV', 'HNW', 'HSG', 'HSH', 'HSJ', 'HSQ', 'HSX', 'HSY', 'HTG', 'HTM', 'HVC', 'IAB', 'IDC', 'IDF', 'IDG', 'IDR', 'IDS', 'IDU', 'IDX', 'IDY', 'IEM', 'IN1', 'IPT', 'ISD', 'ISL', 'ISX', 'IXD', 'J5B', 'JFZ', 'JHM', 'JLT', 'JRV', 'JSV', 'JV4', 'JVA', 'JVS', 'JZR', 'K5B', 'K99', 'KBA', 'KBG', 'KD5', 'KDA', 'KDB', 'KDD', 'KDE', 'KDF', 'KDM', 'KDN', 'KDO', 'KDR', 'KFN', 'KG1', 'KGM', 'KHP', 'KME', 'KO1', 'KO2', 'KOT', 'KTU', 'L0W', 'L1L', 'L6S', 'L6T', 'LAG', 'LAH', 'LAI', 'LAK', 'LAO', 'LAT', 'LB2', 'LBS', 'LBT', 'LCN', 'LDY', 'LEC', 'LER', 'LFC', 'LFR', 'LGC', 'LGU', 'LKA', 'LKS', 'LM2', 'LMO', 'LNV', 'LOG', 'LOX', 'LRH', 'LTG', 'LVO', 'LVZ', 'LXB', 'LXC', 'LXZ', 'LZ0', 'M1F', 'M1P', 'M2F', 'M3M', 'M3N', 'M55', 'M6D', 'M6P', 'M7B', 'M7P', 'M8C', 'MA1', 'MA2', 'MA3', 'MA8', 'MAB', 'MAF', 'MAG', 'MAL', 'MAN', 'MAT', 'MAV', 'MAW', 'MBE', 'MBF', 'MBG', 'MCU', 'MDA', 'MDP', 'MFB', 'MFU', 'MG5', 'MGC', 'MGL', 'MGS', 'MJJ', 'MLB', 'MLR', 'MMA', 'MN0', 'MNA', 'MQG', 'MQT', 'MRH', 'MRP', 'MSX', 'MTT', 'MUB', 'MUR', 'MVP', 'MXY', 'MXZ', 'MYG', 'N1L', 'N3U', 'N9S', 'NA1', 'NAA', 'NAG', 'NBG', 'NBX', 'NBY', 'NDG', 'NFG', 'NG1', 'NG6', 'NGA', 'NGC', 'NGE', 'NGK', 'NGR', 'NGS', 'NGY', 'NGZ', 'NHF', 'NLC', 'NM6', 'NM9', 'NNG', 'NPF', 'NSQ', 'NT1', 'NTF', 'NTO', 'NTP', 'NXD', 'NYT', 'OAK', 'OI7', 'OPM', 'OSU', 'OTG', 'OTN', 'OTU', 'OX2', 'P53', 'P6P', 'P8E', 'PA1', 'PAV', 'PDX', 'PH5', 'PKM', 'PNA', 'PNG', 'PNJ', 'PNW', 'PPC', 'PRP', 'PSG', 'PSV', 'PTQ', 'PUF', 'PZU', 'QDK', 'QIF', 'QKH', 'QPS', 'QV4', 'R1P', 'R1X', 'R2B', 'R2G', 'RAE', 'RAF', 'RAM', 'RAO', 'RB5', 'RBL', 'RCD', 'RER', 'RF5', 'RG1', 'RGG', 'RHA', 'RHC', 'RI2', 'RIB', 'RIP', 'RM4', 'RP3', 'RP5', 'RP6', 'RR7', 'RRJ', 'RRY', 'RST', 'RTG', 'RTV', 'RUG', 'RUU', 'RV7', 'RVG', 'RVM', 'RWI', 'RY7', 'RZM', 'S7P', 'S81', 'SA0', 'SCG', 'SCR', 'SDY', 'SEJ', 'SF6', 'SF9', 'SFU', 'SG4', 'SG5', 'SG6', 'SG7', 'SGA', 'SGC', 'SGD', 'SGN', 'SHB', 'SHD', 'SHG', 'SIA', 'SID', 'SIO', 'SIZ', 'SLB', 'SLM', 'SLT', 'SMD', 'SN5', 'SNG', 'SOE', 'SOG', 'SOL', 'SOR', 'SR1', 'SSG', 'SSH', 'STW', 'STZ', 'SUC', 'SUP', 'SUS', 'SWE', 'SZZ', 'T68', 'T6D', 'T6P', 'T6T', 'TA6', 'TAG', 'TCB', 'TDG', 'TEU', 'TF0', 'TFU', 'TGA', 'TGK', 'TGR', 'TGY', 'TH1', 'TM5', 'TM6', 'TMR', 'TMX', 'TNX', 'TOA', 'TOC', 'TQY', 'TRE', 'TRV', 'TS8', 'TT7', 'TTV', 'TU4', 'TUG', 'TUJ', 'TUP', 'TUR', 'TVD', 'TVG', 'TVM', 'TVS', 'TVV', 'TVY', 'TW7', 'TWA', 'TWD', 'TWG', 'TWJ', 'TWY', 'TXB', 'TYV', 'U1Y', 'U2A', 'U2D', 'U63', 'U8V', 'U97', 'U9A', 'U9D', 'U9G', 'U9J', 'U9M', 'UAP', 'UBH', 'UBO', 'UDC', 'UEA', 'V3M', 'V3P', 'V71', 'VG1', 'VJ1', 'VJ4', 'VKN', 'VTB', 'W9T', 'WIA', 'WOO', 'WUN', 'WZ1', 'WZ2', 'X0X', 'X1P', 'X1X', 'X2F', 'X2Y', 'X34', 'X6X', 'X6Y', 'XDX', 'XGP', 'XIL', 'XKJ', 'XLF', 'XLS', 'XMM', 'XS2', 'XXM', 'XXR', 'XXX', 'XYF', 'XYL', 'XYP', 'XYS', 'XYT', 'XYZ', 'YDR', 'YIO', 'YJM', 'YKR', 'YO5', 'YX0', 'YX1', 'YYB', 'YYH', 'YYJ', 'YYK', 'YYM', 'YYQ', 'YZ0', 'Z0F', 'Z15', 'Z16', 'Z2D', 'Z2T', 'Z3K', 'Z3L', 'Z3Q', 'Z3U', 'Z4K', 'Z4R', 'Z4S', 'Z4U', 'Z4V', 'Z4W', 'Z4Y', 'Z57', 'Z5J', 'Z5L', 'Z61', 'Z6H', 'Z6J', 'Z6W', 'Z8H', 'Z8T', 'Z9D', 'Z9E', 'Z9H', 'Z9K', 'Z9L', 'Z9M', 'Z9N', 'Z9W', 'ZB0', 'ZB1', 'ZB2', 'ZB3', 'ZCD', 'ZCZ', 'ZD0', 'ZDC', 'ZDO', 'ZEE', 'ZEL', 'ZGE', 'ZMR'}
IONS = {'118', '119', '1AL', '1CU', '2FK', '2HP', '2OF', '3CO', '3MT', '3NI', '3OF', '4MO', '4PU', '4TI', '543', '6MO', 'AG', 'AL', 'ALF', 'AM', 'ATH', 'AU', 'AU3', 'AUC', 'BA', 'BEF', 'BF4', 'BO4', 'BR', 'BS3', 'BSY', 'CA', 'CAC', 'CD', 'CD1', 'CD3', 'CD5', 'CE', 'CF', 'CHT', 'CO', 'CO5', 'CON', 'CR', 'CS', 'CSB', 'CU', 'CU1', 'CU2', 'CU3', 'CUA', 'CUZ', 'CYN', 'DME', 'DMI', 'DSC', 'DTI', 'DY', 'E4N', 'EDR', 'EMC', 'ER3', 'EU', 'EU3', 'F', 'FE', 'FE2', 'FPO', 'GA', 'GD3', 'GEP', 'HAI', 'HG', 'HGC', 'HO3', 'IN', 'IR', 'IR3', 'IRI', 'IUM', 'K', 'KO4', 'LA', 'LCO', 'LCP', 'LI', 'LU', 'MAC', 'MG', 'MH2', 'MH3', 'MMC', 'MN', 'MN3', 'MN5', 'MN6', 'MO', 'MO1', 'MO2', 'MO3', 'MO4', 'MO5', 'MO6', 'MOO', 'MOS', 'MOW', 'MW1', 'MW2', 'MW3', 'NA2', 'NA5', 'NA6', 'NAO', 'NAW', 'NET', 'NI', 'NI1', 'NI2', 'NI3', 'NO2', 'NRU', 'O4M', 'OAA', 'OC1', 'OC2', 'OC3', 'OC4', 'OC5', 'OC6', 'OC7', 'OC8', 'OCL', 'OCM', 'OCN', 'OCO', 'OF1', 'OF2', 'OF3', 'OH', 'OS', 'OS4', 'OXL', 'PB', 'PBM', 'PD', 'PER', 'PI', 'PO3', 'PR', 'PT', 'PT4', 'PTN', 'RB', 'RH3', 'RHD', 'RU', 'SB', 'SE4', 'SEK', 'SM', 'SMO', 'SO3', 'T1A', 'TB', 'TBA', 'TCN', 'TEA', 'TH', 'THE', 'TL', 'TMA', 'TRA', 'V', 'VN3', 'VO4', 'W', 'WO5', 'Y1', 'YB', 'YB2', 'YH', 'YT3', 'ZCM', 'ZN', 'ZN2', 'ZN3', 'ZNO', 'ZO3', 'ZR'}

EXCEPTIONS |= {'HOH', 'DOD'} # Added by Jianwei Zhu
GLYCANS |= {'L6N', 'XY6', 'XY9', 'YYD', 'YZT', 'Z6G'} # Added by Jianwei Zhu
IONS |= {'ND', 'NT3', 'RHF', 'PDV'} # Added by Jianwei Zhu
STDAAS = {'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL', 'UNK'}
STDNAS = {'A', 'C', 'G', 'U', 'DA', 'DC', 'DG', 'DT', 'N', 'DN'}
BONDORDER = {'SING': 1, 'DOUB': 2, 'TRIP': 3}


def mol2graph(mol) -> dict:
    """
    Converts molecule sdf file to graph Data object. Modified from
    https://github.com/snap-stanford/ogb/blob/master/ogb/utils/mol.py
    :input: molecule_path (str)
    :return: graph object
    """
    # atoms
    atom_features_list = []
    coords = []
    for i, atom in enumerate(mol.GetAtoms()):
        atom_features_list.append(atom_to_feature_vector(atom))
        coords.append(list(mol.GetConformer().GetAtomPosition(i)))
    x = np.array(atom_features_list, dtype = np.int64)

    # bonds
    num_bond_features = 3  # bond type, bond stereo, is_conjugated
    if len(mol.GetBonds()) > 0: # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()

            edge_feature = bond_to_feature_vector(bond)

            # add edges in both directions
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = np.array(edges_list, dtype = np.int64).T

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = np.array(edge_features_list, dtype = np.int64)

    else:   # mol has no bonds
        edge_index = np.empty((2, 0), dtype = np.int64)
        edge_attr = np.empty((0, num_bond_features), dtype = np.int64)

    graph = dict()
    graph['edge_index'] = edge_index
    graph['edge_feat'] = edge_attr
    graph['node_feat'] = x
    graph['num_nodes'] = len(x)
    graph['coords'] = np.array(coords)

    return graph


def check_mmcif_parsing(parsed_obj: MmcifObject) -> bool:
    try:
        # check number of residues in each chain
        for chain_id in sorted(parsed_obj.chain_to_seqres.keys()):
            seqres = parsed_obj.chain_to_seqres[chain_id]
            structure = parsed_obj.seqres_to_structure[chain_id]
            assert len(seqres) == len(structure), (
                f"has different lengths for chain {chain_id}")
        # check chem_comp consistency
        assert check_chem_comps(parsed_info=parsed_obj.raw_string), (
            f"chem_comp are inconsistent in {parsed_obj.file_id}")
        return True
    except Exception as e:
        logging.error(f"parsing result for {parsed_obj.file_id}. {e}")
        return False


def check_chem_comps(parsed_info: MMCIF2Dict) -> bool:
    # chem_comps ids in _chem_comp
    _entries = mmcif_loop_to_list('_chem_comp.', parsed_info)
    chem_comps = set( [_['_chem_comp.id'] for _ in _entries] )
    # all atom comps parsed from _atom_site
    _entries = mmcif_loop_to_list('_atom_site.', parsed_info)
    atom_comps = set( [_['_atom_site.label_comp_id'] for _ in _entries] )
    # polymer comps parsed from _entity_poly_seq
    _entries = mmcif_loop_to_list('_entity_poly_seq.', parsed_info)
    polyseq_comps = set( [_['_entity_poly_seq.mon_id'] for _ in _entries] )
    # polymer comps parsed from _pdbx_poly_seq_scheme
    _entries = mmcif_loop_to_list('_pdbx_poly_seq_scheme.', parsed_info)
    pdbx_polyseq = set( [_['_pdbx_poly_seq_scheme.mon_id'] for _ in _entries] )
    return  atom_comps.issubset(chem_comps) and polyseq_comps == pdbx_polyseq


def split_chem_comp(chem_comp_full_string: str) -> dict:
    chem_comp_strings = {}
    if chem_comp_full_string.startswith('data_'):
        strings = chem_comp_full_string.split('data_')
        del strings[0] # empty string for first element
        for s in strings:
            lines = s.split('\n')
            chem_comp_strings[lines[0]] = 'data_' + s
    return chem_comp_strings


def process_polymer_residue(residue: ResidueAtPosition,
                            atoms: Sequence[AtomCartn]) -> dict:
    """
    Tokenize standard protein, DNA and RNA residues.
    """
    coord = np.full(3, float('nan'), dtype=np.float32)
    full_coords = np.full((len(ATOMORDER), 3), float('nan'), dtype=np.float32)
    token = {'types': residue.name,
             'coords': coord,
             'full_coords': full_coords}
    if not residue.is_missing and residue.name in STDRESIDUES:
        for atom in atoms:
            if atom.type in ('H', 'D'):
                # AlphaFold3 supplementary information section 2.5.4
                # Filtering of bioassemblies: Hydrogens are removed.
                continue
            pos = np.array([atom.x, atom.y, atom.z], dtype=np.float32)
            if atom.name == 'CA' or atom.name == "C1'":
                # AlphaFold3 supplementary information section 2.5.4
                # For each token we also designate a token centre atom,
                # used in various places below:
                # CA for standard amino acids
                # C1' for standard nucleotides
                coord = pos
            if atom.name in RESIDUEATOMS[residue.name]:
                full_coords[ATOMORDER[atom.name], :] = pos
        token['coords'] = coord
        token['full_coords'] = full_coords

    return token


def process_nonstandard_residue(residue: ResidueAtPosition,
                                atoms: Sequence[AtomCartn],
                                chem_comp_string: str,
                                removeHs: bool = True):
    try:
        # read chemical component information
        handle = io.StringIO(chem_comp_string)
        mmcifdict = MMCIF2Dict.MMCIF2Dict(handle)
        chem_comp_atoms = mmcif_loop_to_dict('_chem_comp_atom.',
                                             '_chem_comp_atom.atom_id',
                                             mmcifdict)
        # convert atom name to index
        idx = 0
        name_to_index = {}
        selected_atoms = []
        for atom in atoms:
            if removeHs and atom.type in ('H', 'D'):
                # AlphaFold3 supplementary information section 2.5.4
                # Filtering of bioassemblies: Hydrogens are removed.
                continue
            if atom.name in name_to_index:
                raise ValueError(f"Duplicated name {atom.name} in protein.cif")
            if atom.name not in chem_comp_atoms:
                # AlphaFold3 supplementary information section 2.5.4
                # Filtering of bioassemblies: For residues or small molecules
                # with CCD codes, atoms outside of the CCD code’s defined set
                # of atom names are removed.
                raise ValueError(f"Cannot find {atom.name} in components.cif")
            idx += 1
            name_to_index[atom.name] = idx
            selected_atoms.append(atom)
        num_atoms = len(selected_atoms)
        # Get the bond type from _chem_comp_bond
        bonds = []
        for bond in mmcif_loop_to_list('_chem_comp_bond.', mmcifdict):
            id1 = bond['_chem_comp_bond.atom_id_1']
            id2 = bond['_chem_comp_bond.atom_id_2']
            if id1 in name_to_index and id2 in name_to_index:
                order = BONDORDER.get(bond['_chem_comp_bond.value_order'], 4)
                if name_to_index[id1] < name_to_index[id2]:
                    index1, index2 = name_to_index[id1], name_to_index[id2]
                else:
                    index1, index2 = name_to_index[id2], name_to_index[id1]
                bonds.append( (index1, index2, order) )
        num_bonds = len(bonds)
        # Write molecule into sdf file, and description from
        # https://www.nonlinear.com/progenesis/sdf-studio/v0.9/faq/sdf-file-format-guidance.aspx
        block = [f'{residue.name}\n'
                 f'  Custom\n'
                 f'\n'
                 f'{num_atoms:3}{num_bonds:3}  0  0  0  0  0  0  0  0  0\n']
        for atom in selected_atoms:
            charge = chem_comp_atoms[atom.name]['_chem_comp_atom.charge']
            charge = charge if charge != '?' else '0'
            block.append(f'{atom.x:10.4f}{atom.y:10.4f}{atom.z:10.4f} '
                         f'{atom.type:<3s} 0{charge:>3s}'
                         f'  0  0  0  0  0  0  0  0  0  0\n')
        for id1, id2, order in sorted(bonds, key=lambda x: x[0]):
            block.append(f'{id1:>3d}{id2:>3d}{order:>3d}  0  0  0  0\n')
        block.append('M  END\n')
        # Ignore RDKit Warnings (https://github.com/rdkit/rdkit/issues/2683)
        # Explicit Valence Error - Partial Sanitization (https://www.rdkit.org/docs/Cookbook.html#explicit-valence-error-partial-sanitization)
        # Pre-condition Violation (https://github.com/rdkit/rdkit/issues/1596)
        RDLogger.DisableLog('rdApp.warning')
        mol = Chem.MolFromMolBlock( ''.join(block), sanitize=False)
        mol.UpdatePropertyCache(strict=False)
        mol = mol if removeHs else Chem.AddHs(mol)
        if not mol or mol.GetNumAtoms() <= 0:
            raise ValueError(f"Invalid rdkit mol {residue.name:3}")
        graph = mol2graph(mol)
        return graph, ''
    except Exception as e:
        return {}, e


def parse_mmcif_string(mmcif_path: str) -> str:
    mmcif_string = ''
    if mmcif_path.endswith('.cif'):
        with open(mmcif_path, 'r') as fp:
            mmcif_string = fp.read()
    elif mmcif_path.endswith('.cif.gz'):
        with gzip.open(mmcif_path, 'rt') as fp:
            mmcif_string = fp.read()
    else:
        logging.error(f"{mmcif_path} must endswith .cif or .cif.gz.")
    return mmcif_string


def process_pdb_complex(mmcif_path: str,
                        chem_comp_path: str,
                        removeHs: bool = True):
    try:
        chem_comp_full_string = parse_mmcif_string(chem_comp_path)
        assert chem_comp_full_string, f"Failed to read comp {chem_comp_path}."
        chem_comp_strings = split_chem_comp(chem_comp_full_string)
        assert chem_comp_strings, f"Failed to parse {chem_comp_path}."
        pdbid = Path(mmcif_path).name.split('.')[0]
        assert len(pdbid) == 4, f"Invalid 4 characters pdbid for {mmcif_path}."
        mmcif_string = parse_mmcif_string(mmcif_path)
        assert mmcif_string, f"Failed to read protein {mmcif_path}."
        res = parse_structure(file_id=pdbid, mmcif_string=mmcif_string) # parse mmcif file by modified AlphaFold mmcif_parsing.py
        assert res.mmcif_object and check_mmcif_parsing(res.mmcif_object), (
            f"Wrong parsing result for pdb {pdbid}")

        # process all chains in complex one by one
        processed_data = {}
        for chain_id in sorted(res.mmcif_object.chain_to_seqres.keys()):
            seqres = res.mmcif_object.chain_to_seqres[chain_id]
            restype = res.mmcif_object.chain_to_restype[chain_id]
            struct = res.mmcif_object.seqres_to_structure[chain_id]
            if len(seqres) - seqres.count('?') < 4:
                # AlphaFold3 supplementary information section 2.5.4
                # Filtering of targets: polymer chain containing fewer than 4
                # resolved residues is filtered out
                continue
            current_chain = {}
            # print('-'*80, f"{pdbid}_{chain_id}", seqres, restype, sep='\n')
            # process residues one by one
            polymer_tokens = []
            non_std_resnas = []
            non_std_graphs = []
            for letter, (residue, atoms) in zip(seqres, struct):
                if letter != '?':
                    # Process polymer residues for protein, DNA and RNA.
                    # Must do this no matter it is standard residue or not.
                    token = process_polymer_residue(residue, atoms)
                    polymer_tokens.append(token)
                if residue.name not in (STDRESIDUES | EXCEPTIONS):
                    # Process non-standard residues for polymer and non-polymer.
                    if not atoms: # pass residue without atoms
                        logging.warning(f"{pdbid}_{chain_id} non-std residue "
                                        f"'{residue.name:3}' has no atoms in "
                                        f"{mmcif_path}.")
                        continue
                    graph, error = process_nonstandard_residue(
                        residue, atoms, chem_comp_strings[residue.name], removeHs)
                    if error:
                        logging.warning(f"{pdbid}_{chain_id} non-std residue "
                                        f"'{residue.name:3}' parse failed in "
                                        f"{mmcif_path}. {error}")
                    else:
                        non_std_resnas.append(residue.name)
                        non_std_graphs.append(graph)
            # post-process polymer
            polymer = {}
            if not polymer_tokens:
                # AlphaFold3 supplementary information section 2.5.4
                # Filtering of bioassemblies: Polymer chains with all unknown
                # residues are removed.
                logging.warning(f"{pdbid}_{chain_id} no polymer {mmcif_path}.")
                continue
            for key in {'types', 'coords', 'full_coords'}:
                polymer[key] = np.stack([_[key] for _ in polymer_tokens])
            polymer['num_residues'] = len(polymer['types'])
            polymer['polyseq'] = np.array([_ for _ in seqres if _ != '?'])
            polymer['polymer_type'] = 'peptide'
            if restype.count('p') < restype.count('n'):
                polymer['polymer_type'] = 'nucleotide'
            current_chain['polymer'] = polymer
            # post-process non-polymer
            for i, (n, g) in enumerate( zip(non_std_resnas, non_std_graphs) ):
                current_chain[f'mol_{i}_{n}'] = g
            # collect current chain data
            processed_data[chain_id] = current_chain
        assert processed_data, f"{mmcif_path} has no desirable chains."
        logging.info(f"{mmcif_path} processing success.")
        return pdbid, res.mmcif_object.header, processed_data
    except Exception as e:
        logging.error(f"{mmcif_path} processing failed. {e}")
        return '', {}, {}


def show_one_complex(processed_data: dict):
    for chain_id, chain_data in processed_data.items():
        print("-"*80, f"Chain_{chain_id}:", sep='\n')

        polymer = chain_data['polymer']
        print(f"  polymer: type={polymer['polymer_type']}", end= ' ')
        print(f"num_residues={polymer['num_residues']}", end=' ')
        for key, value in chain_data['polymer'].items():
            if key != 'num_residues' and key != 'polymer_type':
                print(f"{key}={value.shape}", end=' ')
        print()
        arr = polymer['polyseq']
        print(f"    polyseq[:]      : {''.join(arr)}")
        arr = [f'{_:s}' for _ in polymer['types'][:10]]
        print(f"    types[:10]      : [{', '.join(arr)}]")
        arr = [f'{_:.3f}' for _ in polymer['coords'][:10, 0]]
        print(f"    coords[:10,0]   : [{', '.join(arr)}]")
        arr = [f'{_:.3f}' for _ in polymer['full_coords'][:10, 0, 0]]
        print(f"    full_co[:10,0,0]: [{', '.join(arr)}]")

        for key, value in chain_data.items():
            if key == 'polymer': continue
            print(f"  {key}: num_nodes={value['num_nodes']}", end=' ')
            for k, v in value.items():
                k != 'num_nodes' and print(f"{k}={v.shape}", end=' ')
            print()
            arr = [f'{_}' for _ in value['node_feat'][:10, 0]]
            print(f"    atoms[:10]      : [{', '.join(arr)}]")
            arr = [f'{_:.3f}' for _ in value['coords'][:10, 0]]
            print(f"    coords[:10,0]   : [{', '.join(arr)}]")
            print("    node_feat[:3,:] :", value['node_feat'][:3, :].tolist())
            print("    edge_inde[:,:10]:", value['edge_index'][:, :10].tolist())
            print("    edge_feat[:10,:]:", value['edge_feat'][:10, :].tolist())


if __name__ == '__main__':
    if len(sys.argv) != 3 and len(sys.argv) != 4:
        sys.exit(f"Usage: {sys.argv[0]} <input_mmcif_path> <chem_comp_path> [--keepHs]")
    mmcif_path, chem_comp_path = sys.argv[1:3]
    removeHs = False if len(sys.argv) == 4 and sys.argv[3] == '--keepHs' else True

    id, header, data = process_pdb_complex(mmcif_path, chem_comp_path, removeHs)
    print('-'*80, mmcif_path, chem_comp_path, id, header, sep='\n')
    show_one_complex(data)
