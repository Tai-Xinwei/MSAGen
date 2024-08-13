#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import gzip
import io
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence, Tuple, Union

import click
import lmdb
import numpy as np
from absl import logging
from Bio.PDB import MMCIF2Dict
from joblib import delayed, Parallel
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm

from commons import bstr2obj, obj2bstr
from mmcif_parsing import AtomCartn
from mmcif_parsing import mmcif_loop_to_list
from mmcif_parsing import parse_structure
from mmcif_parsing import ResidueAtPosition
from residue_constants import ATOMORDER, RESIDUEATOMS

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from sfm.data.mol_data.utils.molecule import mol2graph


logging.set_verbosity(logging.INFO)


# From AlphaFold 3 Appendix: CCD code and PDB ID tables
STDRES = {'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL', 'UNK', 'A', 'C', 'G', 'U', 'DA', 'DC', 'DG', 'DT', 'N', 'DN'}
EXCLUS = {'144', '15P', '1PE', '2F2', '2JC', '3HR', '3SY', '7N5', '7PE', '9JE', 'AAE', 'ABA', 'ACE', 'ACN', 'ACT', 'ACY', 'AZI', 'BAM', 'BCN', 'BCT', 'BDN', 'BEN', 'BME', 'BO3', 'BTB', 'BTC', 'BU1', 'C8E', 'CAD', 'CAQ', 'CBM', 'CCN', 'CIT', 'CL', 'CLR', 'CM', 'CMO', 'CO3', 'CPT', 'CXS', 'D10', 'DEP', 'DIO', 'DMS', 'DN', 'DOD', 'DOX', 'EDO', 'EEE', 'EGL', 'EOH', 'EOX', 'EPE', 'ETF', 'FCY', 'FJO', 'FLC', 'FMT', 'FW5', 'GOL', 'GSH', 'GTT', 'GYF', 'HED', 'IHP', 'IHS', 'IMD', 'IOD', 'IPA', 'IPH', 'LDA', 'MB3', 'MEG', 'MES', 'MLA', 'MLI', 'MOH', 'MPD', 'MRD', 'MSE', 'MYR', 'N', 'NA', 'NH2', 'NH4', 'NHE', 'NO3', 'O4B', 'OHE', 'OLA', 'OLC', 'OMB', 'OME', 'OXA', 'P6G', 'PE3', 'PE4', 'PEG', 'PEO', 'PEP', 'PG0', 'PG4', 'PGE', 'PGR', 'PLM', 'PO4', 'POL', 'POP', 'PVO', 'SAR', 'SCN', 'SEO', 'SEP', 'SIN', 'SO4', 'SPD', 'SPM', 'SR', 'STE', 'STO', 'STU', 'TAR', 'TBU', 'TME', 'TPO', 'TRS', 'UNK', 'UNL', 'UNX', 'UPL', 'URE'}
GLYCAN = {'045', '05L', '07E', '07Y', '08U', '09X', '0BD', '0H0', '0HX', '0LP', '0MK', '0NZ', '0UB', '0V4', '0WK', '0XY', '0YT', '10M', '12E', '145', '147', '149', '14T', '15L', '16F', '16G', '16O', '17T', '18D', '18O', '1CF', '1FT', '1GL', '1GN', '1LL', '1S3', '1S4', '1SD', '1X4', '20S', '20X', '22O', '22S', '23V', '24S', '25E', '26O', '27C', '289', '291', '293', '2DG', '2DR', '2F8', '2FG', '2FL', '2GL', '2GS', '2H5', '2HA', '2M4', '2M5', '2M8', '2OS', '2WP', '2WS', '32O', '34V', '38J', '3BU', '3DO', '3DY', '3FM', '3GR', '3HD', '3J3', '3J4', '3LJ', '3LR', '3MG', '3MK', '3R3', '3S6', '3SA', '3YW', '40J', '42D', '445', '44S', '46D', '46Z', '475', '48Z', '491', '49A', '49S', '49T', '49V', '4AM', '4CQ', '4GC', '4GL', '4GP', '4JA', '4N2', '4NN', '4QY', '4R1', '4RS', '4SG', '4UZ', '4V5', '50A', '51N', '56N', '57S', '5GF', '5GO', '5II', '5KQ', '5KS', '5KT', '5KV', '5L3', '5LS', '5LT', '5MM', '5N6', '5QP', '5SP', '5TH', '5TJ', '5TK', '5TM', '61J', '62I', '64K', '66O', '6BG', '6C2', '6DM', '6GB', '6GP', '6GR', '6K3', '6KH', '6KL', '6KS', '6KU', '6KW', '6LA', '6LS', '6LW', '6MJ', '6MN', '6PZ', '6S2', '6UD', '6YR', '6ZC', '73E', '79J', '7CV', '7D1', '7GP', '7JZ', '7K2', '7K3', '7NU', '83Y', '89Y', '8B7', '8B9', '8EX', '8GA', '8GG', '8GP', '8I4', '8LR', '8OQ', '8PK', '8S0', '8YV', '95Z', '96O', '98U', '9AM', '9C1', '9CD', '9GP', '9KJ', '9MR', '9OK', '9PG', '9QG', '9S7', '9SG', '9SJ', '9SM', '9SP', '9T1', '9T7', '9VP', '9WJ', '9WN', '9WZ', '9YW', 'A0K', 'A1Q', 'A2G', 'A5C', 'A6P', 'AAL', 'ABD', 'ABE', 'ABF', 'ABL', 'AC1', 'ACR', 'ACX', 'ADA', 'AF1', 'AFD', 'AFO', 'AFP', 'AGL', 'AH2', 'AH8', 'AHG', 'AHM', 'AHR', 'AIG', 'ALL', 'ALX', 'AMG', 'AMN', 'AMU', 'AMV', 'ANA', 'AOG', 'AQA', 'ARA', 'ARB', 'ARI', 'ARW', 'ASC', 'ASG', 'ASO', 'AXP', 'AXR', 'AY9', 'AZC', 'B0D', 'B16', 'B1H', 'B1N', 'B2G', 'B4G', 'B6D', 'B7G', 'B8D', 'B9D', 'BBK', 'BBV', 'BCD', 'BDF', 'BDG', 'BDP', 'BDR', 'BEM', 'BFN', 'BG6', 'BG8', 'BGC', 'BGL', 'BGN', 'BGP', 'BGS', 'BHG', 'BM3', 'BM7', 'BMA', 'BMX', 'BND', 'BNG', 'BNX', 'BO1', 'BOG', 'BQY', 'BS7', 'BTG', 'BTU', 'BW3', 'BWG', 'BXF', 'BXP', 'BXX', 'BXY', 'BZD', 'C3B', 'C3G', 'C3X', 'C4B', 'C4W', 'C5X', 'CBF', 'CBI', 'CBK', 'CDR', 'CE5', 'CE6', 'CE8', 'CEG', 'CEZ', 'CGF', 'CJB', 'CKB', 'CKP', 'CNP', 'CR1', 'CR6', 'CRA', 'CT3', 'CTO', 'CTR', 'CTT', 'D1M', 'D5E', 'D6G', 'DAF', 'DAG', 'DAN', 'DDA', 'DDL', 'DEG', 'DEL', 'DFR', 'DFX', 'DG0', 'DGO', 'DGS', 'DGU', 'DJB', 'DJE', 'DK4', 'DKX', 'DKZ', 'DL6', 'DLD', 'DLF', 'DLG', 'DNO', 'DO8', 'DOM', 'DPC', 'DQR', 'DR2', 'DR3', 'DR5', 'DRI', 'DSR', 'DT6', 'DVC', 'DYM', 'E3M', 'E5G', 'EAG', 'EBG', 'EBQ', 'EEN', 'EEQ', 'EGA', 'EMP', 'EMZ', 'EPG', 'EQP', 'EQV', 'ERE', 'ERI', 'ETT', 'EUS', 'F1P', 'F1X', 'F55', 'F58', 'F6P', 'F8X', 'FBP', 'FCA', 'FCB', 'FCT', 'FDP', 'FDQ', 'FFC', 'FFX', 'FIF', 'FK9', 'FKD', 'FMF', 'FMO', 'FNG', 'FNY', 'FRU', 'FSA', 'FSI', 'FSM', 'FSW', 'FUB', 'FUC', 'FUD', 'FUF', 'FUL', 'FUY', 'FVQ', 'FX1', 'FYJ', 'G0S', 'G16', 'G1P', 'G20', 'G28', 'G2F', 'G3F', 'G3I', 'G4D', 'G4S', 'G6D', 'G6P', 'G6S', 'G7P', 'G8Z', 'GAA', 'GAC', 'GAD', 'GAF', 'GAL', 'GAT', 'GBH', 'GC1', 'GC4', 'GC9', 'GCB', 'GCD', 'GCN', 'GCO', 'GCS', 'GCT', 'GCU', 'GCV', 'GCW', 'GDA', 'GDL', 'GE1', 'GE3', 'GFP', 'GIV', 'GL0', 'GL1', 'GL2', 'GL4', 'GL5', 'GL6', 'GL7', 'GL9', 'GLA', 'GLC', 'GLD', 'GLF', 'GLG', 'GLO', 'GLP', 'GLS', 'GLT', 'GM0', 'GMB', 'GMH', 'GMT', 'GMZ', 'GN1', 'GN4', 'GNS', 'GNX', 'GP0', 'GP1', 'GP4', 'GPH', 'GPK', 'GPM', 'GPO', 'GPQ', 'GPU', 'GPV', 'GPW', 'GQ1', 'GRF', 'GRX', 'GS1', 'GS9', 'GTK', 'GTM', 'GTR', 'GU0', 'GU1', 'GU2', 'GU3', 'GU4', 'GU5', 'GU6', 'GU8', 'GU9', 'GUF', 'GUL', 'GUP', 'GUZ', 'GXL', 'GXV', 'GYE', 'GYG', 'GYP', 'GYU', 'GYV', 'GZL', 'H1M', 'H1S', 'H2P', 'H3S', 'H53', 'H6Q', 'H6Z', 'HBZ', 'HD4', 'HNV', 'HNW', 'HSG', 'HSH', 'HSJ', 'HSQ', 'HSX', 'HSY', 'HTG', 'HTM', 'HVC', 'IAB', 'IDC', 'IDF', 'IDG', 'IDR', 'IDS', 'IDU', 'IDX', 'IDY', 'IEM', 'IN1', 'IPT', 'ISD', 'ISL', 'ISX', 'IXD', 'J5B', 'JFZ', 'JHM', 'JLT', 'JRV', 'JSV', 'JV4', 'JVA', 'JVS', 'JZR', 'K5B', 'K99', 'KBA', 'KBG', 'KD5', 'KDA', 'KDB', 'KDD', 'KDE', 'KDF', 'KDM', 'KDN', 'KDO', 'KDR', 'KFN', 'KG1', 'KGM', 'KHP', 'KME', 'KO1', 'KO2', 'KOT', 'KTU', 'L0W', 'L1L', 'L6S', 'L6T', 'LAG', 'LAH', 'LAI', 'LAK', 'LAO', 'LAT', 'LB2', 'LBS', 'LBT', 'LCN', 'LDY', 'LEC', 'LER', 'LFC', 'LFR', 'LGC', 'LGU', 'LKA', 'LKS', 'LM2', 'LMO', 'LNV', 'LOG', 'LOX', 'LRH', 'LTG', 'LVO', 'LVZ', 'LXB', 'LXC', 'LXZ', 'LZ0', 'M1F', 'M1P', 'M2F', 'M3M', 'M3N', 'M55', 'M6D', 'M6P', 'M7B', 'M7P', 'M8C', 'MA1', 'MA2', 'MA3', 'MA8', 'MAB', 'MAF', 'MAG', 'MAL', 'MAN', 'MAT', 'MAV', 'MAW', 'MBE', 'MBF', 'MBG', 'MCU', 'MDA', 'MDP', 'MFB', 'MFU', 'MG5', 'MGC', 'MGL', 'MGS', 'MJJ', 'MLB', 'MLR', 'MMA', 'MN0', 'MNA', 'MQG', 'MQT', 'MRH', 'MRP', 'MSX', 'MTT', 'MUB', 'MUR', 'MVP', 'MXY', 'MXZ', 'MYG', 'N1L', 'N3U', 'N9S', 'NA1', 'NAA', 'NAG', 'NBG', 'NBX', 'NBY', 'NDG', 'NFG', 'NG1', 'NG6', 'NGA', 'NGC', 'NGE', 'NGK', 'NGR', 'NGS', 'NGY', 'NGZ', 'NHF', 'NLC', 'NM6', 'NM9', 'NNG', 'NPF', 'NSQ', 'NT1', 'NTF', 'NTO', 'NTP', 'NXD', 'NYT', 'OAK', 'OI7', 'OPM', 'OSU', 'OTG', 'OTN', 'OTU', 'OX2', 'P53', 'P6P', 'P8E', 'PA1', 'PAV', 'PDX', 'PH5', 'PKM', 'PNA', 'PNG', 'PNJ', 'PNW', 'PPC', 'PRP', 'PSG', 'PSV', 'PTQ', 'PUF', 'PZU', 'QDK', 'QIF', 'QKH', 'QPS', 'QV4', 'R1P', 'R1X', 'R2B', 'R2G', 'RAE', 'RAF', 'RAM', 'RAO', 'RB5', 'RBL', 'RCD', 'RER', 'RF5', 'RG1', 'RGG', 'RHA', 'RHC', 'RI2', 'RIB', 'RIP', 'RM4', 'RP3', 'RP5', 'RP6', 'RR7', 'RRJ', 'RRY', 'RST', 'RTG', 'RTV', 'RUG', 'RUU', 'RV7', 'RVG', 'RVM', 'RWI', 'RY7', 'RZM', 'S7P', 'S81', 'SA0', 'SCG', 'SCR', 'SDY', 'SEJ', 'SF6', 'SF9', 'SFU', 'SG4', 'SG5', 'SG6', 'SG7', 'SGA', 'SGC', 'SGD', 'SGN', 'SHB', 'SHD', 'SHG', 'SIA', 'SID', 'SIO', 'SIZ', 'SLB', 'SLM', 'SLT', 'SMD', 'SN5', 'SNG', 'SOE', 'SOG', 'SOL', 'SOR', 'SR1', 'SSG', 'SSH', 'STW', 'STZ', 'SUC', 'SUP', 'SUS', 'SWE', 'SZZ', 'T68', 'T6D', 'T6P', 'T6T', 'TA6', 'TAG', 'TCB', 'TDG', 'TEU', 'TF0', 'TFU', 'TGA', 'TGK', 'TGR', 'TGY', 'TH1', 'TM5', 'TM6', 'TMR', 'TMX', 'TNX', 'TOA', 'TOC', 'TQY', 'TRE', 'TRV', 'TS8', 'TT7', 'TTV', 'TU4', 'TUG', 'TUJ', 'TUP', 'TUR', 'TVD', 'TVG', 'TVM', 'TVS', 'TVV', 'TVY', 'TW7', 'TWA', 'TWD', 'TWG', 'TWJ', 'TWY', 'TXB', 'TYV', 'U1Y', 'U2A', 'U2D', 'U63', 'U8V', 'U97', 'U9A', 'U9D', 'U9G', 'U9J', 'U9M', 'UAP', 'UBH', 'UBO', 'UDC', 'UEA', 'V3M', 'V3P', 'V71', 'VG1', 'VJ1', 'VJ4', 'VKN', 'VTB', 'W9T', 'WIA', 'WOO', 'WUN', 'WZ1', 'WZ2', 'X0X', 'X1P', 'X1X', 'X2F', 'X2Y', 'X34', 'X6X', 'X6Y', 'XDX', 'XGP', 'XIL', 'XKJ', 'XLF', 'XLS', 'XMM', 'XS2', 'XXM', 'XXR', 'XXX', 'XYF', 'XYL', 'XYP', 'XYS', 'XYT', 'XYZ', 'YDR', 'YIO', 'YJM', 'YKR', 'YO5', 'YX0', 'YX1', 'YYB', 'YYH', 'YYJ', 'YYK', 'YYM', 'YYQ', 'YZ0', 'Z0F', 'Z15', 'Z16', 'Z2D', 'Z2T', 'Z3K', 'Z3L', 'Z3Q', 'Z3U', 'Z4K', 'Z4R', 'Z4S', 'Z4U', 'Z4V', 'Z4W', 'Z4Y', 'Z57', 'Z5J', 'Z5L', 'Z61', 'Z6H', 'Z6J', 'Z6W', 'Z8H', 'Z8T', 'Z9D', 'Z9E', 'Z9H', 'Z9K', 'Z9L', 'Z9M', 'Z9N', 'Z9W', 'ZB0', 'ZB1', 'ZB2', 'ZB3', 'ZCD', 'ZCZ', 'ZD0', 'ZDC', 'ZDO', 'ZEE', 'ZEL', 'ZGE', 'ZMR'}
IONCCD = {'118', '119', '1AL', '1CU', '2FK', '2HP', '2OF', '3CO', '3MT', '3NI', '3OF', '4MO', '4PU', '4TI', '543', '6MO', 'AG', 'AL', 'ALF', 'AM', 'ATH', 'AU', 'AU3', 'AUC', 'BA', 'BEF', 'BF4', 'BO4', 'BR', 'BS3', 'BSY', 'CA', 'CAC', 'CD', 'CD1', 'CD3', 'CD5', 'CE', 'CF', 'CHT', 'CO', 'CO5', 'CON', 'CR', 'CS', 'CSB', 'CU', 'CU1', 'CU2', 'CU3', 'CUA', 'CUZ', 'CYN', 'DME', 'DMI', 'DSC', 'DTI', 'DY', 'E4N', 'EDR', 'EMC', 'ER3', 'EU', 'EU3', 'F', 'FE', 'FE2', 'FPO', 'GA', 'GD3', 'GEP', 'HAI', 'HG', 'HGC', 'HO3', 'IN', 'IR', 'IR3', 'IRI', 'IUM', 'K', 'KO4', 'LA', 'LCO', 'LCP', 'LI', 'LU', 'MAC', 'MG', 'MH2', 'MH3', 'MMC', 'MN', 'MN3', 'MN5', 'MN6', 'MO', 'MO1', 'MO2', 'MO3', 'MO4', 'MO5', 'MO6', 'MOO', 'MOS', 'MOW', 'MW1', 'MW2', 'MW3', 'NA2', 'NA5', 'NA6', 'NAO', 'NAW', 'NET', 'NI', 'NI1', 'NI2', 'NI3', 'NO2', 'NRU', 'O4M', 'OAA', 'OC1', 'OC2', 'OC3', 'OC4', 'OC5', 'OC6', 'OC7', 'OC8', 'OCL', 'OCM', 'OCN', 'OCO', 'OF1', 'OF2', 'OF3', 'OH', 'OS', 'OS4', 'OXL', 'PB', 'PBM', 'PD', 'PER', 'PI', 'PO3', 'PR', 'PT', 'PT4', 'PTN', 'RB', 'RH3', 'RHD', 'RU', 'SB', 'SE4', 'SEK', 'SM', 'SMO', 'SO3', 'T1A', 'TB', 'TBA', 'TCN', 'TEA', 'TH', 'THE', 'TL', 'TMA', 'TRA', 'V', 'VN3', 'VO4', 'W', 'WO5', 'Y1', 'YB', 'YB2', 'YH', 'YT3', 'ZCM', 'ZN', 'ZN2', 'ZN3', 'ZNO', 'ZO3', 'ZR'}
# Added by Jianwei Zhu
EXCLUS |= {'HOH', 'DOD', 'WAT', 'CD'}
EXCLUS |= {'0VI', '8P8', 'A9J', 'ASX', 'BF5', 'D3O', 'D8U', 'DUM', 'GLX', 'H9C', 'ND4', 'NWN', 'SPW', 'S5Q', 'TSD', 'USN', 'VOB'} # Maybe wrong mmCIF files
EXCLUS |= {'08T', '0I7', '0MI', '0OD', '10R', '10S', '1KW', '1MK', '1WT', '25X', '25Y', '26E', '2FK', '34B', '39B', '39E', '3JI', '3UQ', '3ZZ', '4A6', '4EX', '4IR', '4LA', '5L1', '6BP', '6ER', '7Q8', '7RZ', '8M0', '8WV', '9JA', '9JJ', '9JM', '9TH', '9UK', 'A1ALJ', 'A1H7J', 'A1H8D', 'A1ICR', 'AOH', 'B1M', 'B8B', 'BBQ', 'BVR', 'CB5', 'CFN', 'COB', 'CWO', 'D0X', 'D6N', 'DAE', 'DAQ', 'DGQ', 'DKE', 'DVT', 'DW1', 'DW2', 'E52', 'EAQ', 'EJ2', 'ELJ', 'FDC', 'FEM', 'FLL', 'FNE', 'FO4', 'GCR', 'GIX', 'GXW', 'GXZ', 'HB1', 'HFW', 'HUJ', 'I8K', 'ICE', 'ICG', 'ICH', 'ICS', 'ICZ', 'IK6', 'IV9', 'IWL', 'IWO', 'J7T', 'J8B', 'JGH', 'JI8', 'JSU', 'K6G', 'K9G', 'KCO', 'KEG', 'KHN', 'KK5', 'KKE', 'KKH', 'KYS', 'KYT', 'LD3', 'M6O', 'M7E', 'ME3', 'MNQ', 'MO7', 'MYW', 'N1B', 'NA2', 'NA5', 'NA6', 'NAO', 'NAW', 'NE5', 'NFC', 'NFV', 'NMQ', 'NMR', 'NT3', 'O1N', 'O93', 'OEC', 'OER', 'OEX', 'OEY', 'ON6', 'ONP', 'OS1', 'OSW', 'OT1', 'OWK', 'OXV', 'OY5', 'OY8', 'OZN', 'P5F', 'P5T', 'P6D', 'P6Q', 'P7H', 'P7Z', 'P82', 'P8B', 'PHF', 'PNQ', 'PQJ', 'Q2Z', 'Q38', 'Q3E', 'Q3H', 'Q3K', 'Q3N', 'Q3Q', 'Q3T', 'Q3W', 'Q4B', 'Q65', 'Q7V', 'QIY', 'QT4', 'R1N', 'R5N', 'R5Q', 'RAX', 'RBN', 'RCS', 'REI', 'REJ', 'REP', 'REQ', 'RIR', 'RTC', 'RU7', 'RUC', 'RUD', 'RUH', 'RUI', 'S18', 'S31', 'S5T', 'S9F', 'SIW', 'SWR', 'T0P', 'TEW', 'U8G', 'UDF', 'UGO', 'UO3', 'UTX', 'UZC', 'V22', 'V9G', 'VA3', 'VAV', 'VFY', 'VI6', 'VL9', 'VOF', 'VPC', 'VSU', 'VTU', 'VTZ', 'WCO', 'WGB', 'WJS', 'WK5', 'WNI', 'WO2', 'WO3', 'WRK', 'WUX', 'WZW', 'X33', 'X3P', 'X5M', 'X5W', 'XC3', 'XCO', 'XCU', 'XZ6', 'Y59', 'Y77', 'YIA', 'YJ6', 'YJK', 'YQ1', 'YQ4', 'ZIV', 'ZJ5', 'ZKG', 'ZPT', 'ZRW', 'ZV2'} # RDKit fail reading
EXCLUS |= {'07D', '0H2', '0KA', '1CL', '1Y8', '2NO', '2PT', '3T3', '402', '4KV', '4WV', '4WW', '4WX', '6ML', '6WF', '72B', '74C', '8CY', '8JU', '8ZR', '9CO', '9S8', '9SQ', '9UX', 'ARS', 'B51', 'BCB', 'BF8', 'BGQ', 'BJ8', 'BRO', 'CFM', 'CH2', 'CLF', 'CLO', 'CLP', 'CU6', 'CUV', 'CYA', 'CYO', 'CZZ', 'DML', 'DW5', 'EL9', 'ER2', 'ETH', 'EXC', 'F3S', 'F4S', 'FDD', 'FLO', 'FS2', 'FS3', 'FS4', 'FS5', 'FSF', 'FSX', 'FU8', 'FV2', 'GAK', 'GFX', 'GK8', 'GTE', 'GXB', 'H', 'H1T', 'H79', 'HEO', 'HME', 'HNN', 'ICA', 'IDO', 'IF6', 'IHW', 'ITM', 'IWZ', 'IX3', 'J7Q', 'J85', 'J8E', 'J9H', 'JCT', 'JQJ', 'JSC', 'JSD', 'JSE', 'JY1', 'KBW', 'L8W', 'LFH', 'LPJ', 'MAP', 'MEO', 'MHM', 'MHX', 'MNH', 'MNR', 'MTN', 'NFS', 'NGN', 'NH', 'NMO', 'NO', 'NYN', 'O', 'OET', 'OL3', 'OL4', 'OL5', 'OLS', 'OX', 'OXO', 'P4J', 'PMR', 'PT7', 'Q61', 'QTR', 'R1B', 'R1F', 'R7A', 'R9H', 'RCY', 'RFQ', 'RPS', 'RQM', 'RRE', 'RXR', 'S', 'S32', 'S3F', 'SE', 'SF3', 'SF4', 'SFO', 'SFS', 'SI0', 'SI7', 'SVP', 'T9T', 'TBY', 'TDJ', 'TE', 'TL', 'TML', 'U0J', 'UFF', 'UJI', 'UJY', 'V1A', 'VHR', 'VQ8', 'VV2', 'VV7', 'WCC', 'XCC', 'XX2', 'YF8', 'YPT', 'ZJZ', 'ZKP'} # SMILES different (lone-pair electron)
EXCLUS |= {'CHL', 'CL0', 'CL1', 'CL2', 'CL7', 'HE5', 'HEG', 'HES'} # RDKit fail to generate conformer
NUM2SYM = {_: Chem.GetPeriodicTable().GetElementSymbol(_+1) for _ in range(118)}
SYM2NUM = {Chem.GetPeriodicTable().GetElementSymbol(_+1): _ for _ in range(118)}


@click.group()
def cli():
    pass


def chunks(lst: list, n: int) -> list:
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


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


def split_chem_comp_full_string(chem_comp_full_string: str) -> Mapping[str, str]:
    chem_comp_strings = {}
    if chem_comp_full_string.startswith('data_'):
        strings = chem_comp_full_string.split('data_')
        del strings[0] # empty string for first element
        for s in strings:
            lines = s.split('\n')
            chem_comp_strings[lines[0]] = 'data_' + s
    return chem_comp_strings


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


def residue2resdict(chain_id: str,
                    seqres: str,
                    restype: str,
                    residue: ResidueAtPosition,
                    atoms: Sequence[AtomCartn],
                    ) -> Mapping[str, Any]:
    resdict = {'chain_id': chain_id,
               'residue_number': float('nan'),
               'insertion_code': ' ',
               'seqres': seqres,
               'restype': restype,
               'name': residue.name,
               'is_missing': residue.is_missing,
               'hetflag' : residue.hetflag,
               'atoms': []}
    if residue.position:
        resdict['chain_id'] = residue.position.chain_id
        resdict['residue_number'] = residue.position.residue_number
        resdict['insertion_code'] = residue.position.insertion_code
    for atom in atoms:
        resdict['atoms'].append({
            'name': atom.name,
            'type': atom.type,
            'x': atom.x,
            'y': atom.y,
            'z': atom.z,
            })
    return resdict


def process_polymer_chain(polymer_chain: Sequence[dict]) -> Mapping[str, Any]:
    """Process one polymer chain."""
    resname, seqres, restype, center_coord, allatom_coord = [], [], [], [], []
    for residue in polymer_chain:
        resname.append(residue['name'])
        seqres.append(residue['seqres'])
        restype.append(residue['restype'])
        c_coord = np.full(3, float('nan'), dtype=np.float32)
        a_coord = np.full((len(ATOMORDER), 3), float('nan'), dtype=np.float32)
        if not residue['is_missing']:
            _key = residue['name'] if residue['name'] in STDRES else 'UNK'
            for atom in residue['atoms']:
                pos = np.array([atom['x'], atom['y'], atom['z']])
                if atom['name'] == "CA" or atom['name'] == "C1'":
                    # AlphaFold3 supplementary information section 2.5.4
                    # For each token we also designate a token centre atom:
                    # CA for standard amino acids
                    # C1' for standard nucleotides
                    c_coord = pos
                # AlphaFold3 supplementary information section 2.5.4
                # Filtering of bioassemblies: Hydrogens are removed.
                # Atoms should not exist in this residue are excluded.
                if atom['name'] in RESIDUEATOMS[_key]:
                    a_coord[ATOMORDER[atom['name']], :] = pos
        center_coord.append(c_coord)
        allatom_coord.append(a_coord)
    data = {
        'resname': np.array(resname),
        'seqres': np.array(seqres),
        'restype': np.array(restype),
        'center_coord': np.array(center_coord, dtype=np.float32),
        'allatom_coord': np.array(allatom_coord, dtype=np.float32),
    }
    return data


def create_rdkitmol(atoms: Sequence[str],
                    charge: int,
                    bond_orders: Sequence[Tuple[int, int, str]],
                    atom_charges: Sequence[int]
                    ) -> Chem.Mol:
    """Create an RDKit molecule using atom types and bond orders."""
    with Chem.RWMol() as mw:
        for a, c in zip(atoms, atom_charges):
            atom = Chem.Atom(a)
            atom.SetFormalCharge(c)
            mw.AddAtom(atom)

        # Ignore RDKit Warnings (https://github.com/rdkit/rdkit/issues/2683)
        # Explicit Valence Error - Partial Sanitization (https://www.rdkit.org/docs/Cookbook.html#explicit-valence-error-partial-sanitization)
        # Pre-condition Violation (https://github.com/rdkit/rdkit/issues/1596)
        mw.UpdatePropertyCache(strict=False)

        for i, j, order in bond_orders:
            if order == 1:
                mw.AddBond(i, j, Chem.BondType.SINGLE)
            elif order == 2:
                mw.AddBond(i, j, Chem.BondType.DOUBLE)
            elif order == 3:
                mw.AddBond(i, j, Chem.BondType.TRIPLE)
            else:
                mw.AddBond(i, j, Chem.BondType.SINGLE)
        Chem.SanitizeMol(mw)

        if Chem.GetFormalCharge(mw) != charge:
            raise ValueError(f"mol charge {Chem.GetFormalCharge(mw)}!={charge}")

        return mw


def chemcomp2graph(chem_comp_string: str) -> Mapping[str, Any]:
    # process ideal chemical component
    handle = io.StringIO(chem_comp_string)
    mmcifdict = MMCIF2Dict.MMCIF2Dict(handle)
    chem_comp_id = mmcifdict['_chem_comp.id'][0]
    pdbx_formal_charge = int(mmcifdict['_chem_comp.pdbx_formal_charge'][0])
    # parse chem_comp_atoms
    id2index = {}
    chem_comp_atoms = []
    for i, atom in enumerate(mmcif_loop_to_list('_chem_comp_atom.', mmcifdict)):
        _id = atom['_chem_comp_atom.atom_id']
        assert _id not in id2index, f"Duplicate atom {_id} in {chem_comp_id}."
        id2index[_id] = i
        _symbol = atom['_chem_comp_atom.type_symbol']
        _charge = atom['_chem_comp_atom.charge']
        _mc = [float('nan'), float('nan'), float('nan')]
        if atom['_chem_comp_atom.model_Cartn_x'] != '?':
            _mc = [float(atom['_chem_comp_atom.model_Cartn_x']),
                   float(atom['_chem_comp_atom.model_Cartn_y']),
                   float(atom['_chem_comp_atom.model_Cartn_z'])]
        _ci = [float('nan'), float('nan'), float('nan')]
        if atom['_chem_comp_atom.pdbx_model_Cartn_x_ideal'] != '?':
            _ci = [float(atom['_chem_comp_atom.pdbx_model_Cartn_x_ideal']),
                   float(atom['_chem_comp_atom.pdbx_model_Cartn_y_ideal']),
                   float(atom['_chem_comp_atom.pdbx_model_Cartn_z_ideal'])]
        chem_comp_atoms.append({
            'id': _id,
            'symbol': _symbol[0].upper() + _symbol[1:].lower(),
            'charge': int(_charge) if _charge != '?' else 0,
            'model_cartn': np.array(_mc),
            'cartn_ideal': np.array(_ci),
            })
    # parse chem_comp_bonds
    atompairs = set()
    chem_comp_bonds = []
    for bond in mmcif_loop_to_list('_chem_comp_bond.', mmcifdict):
        id1 = bond['_chem_comp_bond.atom_id_1']
        id2 = bond['_chem_comp_bond.atom_id_2']
        assert id1 in id2index and id2 in id2index, (
            f"Invalid bond for atom pair ({id1}, {id2}) in {chem_comp_id}.")
        assert (id1, id2) not in atompairs and (id2, id1) not in atompairs, (
            f"Duplicate atom pair ({id1}, {id2}) in {chem_comp_id}.")
        atompairs.add((id1, id2))
        chem_comp_bonds.append({
            'id1': id1,
            'id2': id2,
            'order': bond['_chem_comp_bond.value_order'],
            })
    # parse ideal coordinates
    model_cartn = np.array([_['model_cartn'] for _ in chem_comp_atoms])
    cartn_ideal = np.array([_['cartn_ideal'] for _ in chem_comp_atoms])
    if sum(np.isnan(model_cartn).ravel()) < sum(np.isnan(cartn_ideal).ravel()):
        cartn = np.nan_to_num(model_cartn)
    else:
        cartn = np.nan_to_num(cartn_ideal)
    # extract feature for chem_comp atoms and bonds
    atomids = [_['id'] for _ in chem_comp_atoms]
    symbols = [_['symbol'] for _ in chem_comp_atoms]
    charges = [_['charge'] for _ in chem_comp_atoms]
    BONDORDER = {'SING': 1, 'DOUB': 2, 'TRIP': 3}
    orders = [(id2index[_['id1']], id2index[_['id2']], BONDORDER[_['order']])
              for _ in chem_comp_bonds]
    try:
        # convert atom symbols and bond orders to mol by using RDKit
        rdkitmol = create_rdkitmol(symbols, pdbx_formal_charge, orders, charges)
        # RDKit generate conformers for molecules using ETKDGv3 method
        params = AllChem.ETKDGv3()
        params.randomSeed = 12345
        stat = AllChem.EmbedMolecule(rdkitmol, params)
        if stat == 0:
            cartn = np.array(rdkitmol.GetConformer().GetPositions())
        rdkitmol.RemoveAllConformers()
        graph = {
            'name': chem_comp_id,
            'pdbx_formal_charge': pdbx_formal_charge,
            'atomids': np.array(atomids),
            'symbols': np.array(symbols),
            'charges': np.array(charges),
            'coords': np.array(cartn, dtype=np.float32),
            'orders': np.array(orders),
            'rdkitmol': rdkitmol,
            }
        # use sfm.data.mol_data.utils.molecule.mol2graph to generate mol graph
        # graph update key node_feat, edge_index, edge_feat, num_nodes
        graph.update(mol2graph(rdkitmol))
        return graph
    except Exception as e:
        logging.error(f"Failed to parse {chem_comp_id}, {e}")
        return {}


def process_nonpoly_residue(residue: ResidueAtPosition,
                            atoms: Sequence[AtomCartn],
                            chem_comp_graph: Mapping[str, Any],
                            ) -> Mapping[str, Any]:
    assert residue.name == chem_comp_graph['name'], (
        f"Residue name {residue.name} does not match {chem_comp_graph['id']}.")
    graph = {'chain_id': residue.position.chain_id,
             'residue_number': residue.position.residue_number}
    graph.update(chem_comp_graph)
    node_coord = [[float('nan')]*3 for _ in chem_comp_graph['atomids']]
    id2index = {id: _ for _, id in enumerate(chem_comp_graph['atomids'])}
    for atom in atoms:
        if atom.name not in id2index:
            # AlphaFold3 supplementary information section 2.5.4
            # Filtering of bioassemblies: For residues or small molecules
            # with CCD codes, atoms outside of the CCD code’s defined set
            # of atom names are removed.
            continue
        node_coord[id2index[atom.name]] = [atom.x, atom.y, atom.z]
    graph['node_coord'] = np.array(node_coord, dtype=np.float32)
    return graph


def process_one_mmcif(mmcif_path: str,
                      chem_comp_path: str,
                      ) -> Mapping[str, Union[str, dict, list]]:
    """Parse a mmCIF file and convert data to list of dict."""
    try:
        chem_comp_path = Path(chem_comp_path).resolve()
        chem_comp_full_string = parse_mmcif_string(str(chem_comp_path))
        assert chem_comp_full_string.startswith('data_'), (
            f"Failed to read chem_comp from {chem_comp_path}.")

        chem_comp_strings = split_chem_comp_full_string(chem_comp_full_string)
        assert chem_comp_strings, f"Failed to split {chem_comp_path}."

        mmcif_path = Path(mmcif_path).resolve()
        pdbid = str(mmcif_path.name).split('.')[0]
        assert len(pdbid) == 4, f"Invalid 4 characters PDBID {pdbid}."

        mmcif_string = parse_mmcif_string(str(mmcif_path))
        assert mmcif_string, f"Failed to read mmcif string for {pdbid}."

        # Parse mmcif file by modified AlphaFold mmcif_parsing.py
        result = parse_structure(file_id=pdbid, mmcif_string=mmcif_string)
        assert result.mmcif_object, f"The errors are {result.errors}"

        # process all chains in mmcif one by one
        chem_comp_graphs = {}
        polymer_chains = {}
        nonpoly_graphs = []
        for chain_id in sorted(result.mmcif_object.chain_to_seqres.keys()):
            seqres = result.mmcif_object.chain_to_seqres[chain_id]
            restype = result.mmcif_object.chain_to_restype[chain_id]
            struct = result.mmcif_object.seqres_to_structure[chain_id]
            # print('-'*80, f"{pdbid}_{chain_id}", seqres, restype, sep='\n')
            current_chain = []
            # process residues one by one
            for sr, rt, (residue, atoms) in zip(seqres, restype, struct):
                if residue.position and chain_id != residue.position.chain_id:
                    raise ValueError(f"Chain '{chain_id}' has wrong {residue}")
                if sr != '?': # <=> rt != '*'
                    # Process polymer residues for protein, DNA and RNA.
                    # Must do this no matter it is standard residue or not.
                    current_chain.append(
                        residue2resdict(chain_id, sr, rt, residue, atoms))
                else: # sr == '?': nonpoly residues
                    if residue.name in (EXCLUS | GLYCAN) or not atoms:
                        # Process non-polymer residues
                        # Skip non-standard residue without any atoms
                        continue
                    if residue.name not in chem_comp_graphs:
                        chem_comp_graphs[residue.name] = chemcomp2graph(
                            chem_comp_strings[residue.name])
                    _graph = chem_comp_graphs[residue.name]
                    nonpoly_graphs.append(
                        process_nonpoly_residue(residue, atoms, _graph))
            if len(current_chain) >= 4:
                # AlphaFold3 supplementary information section 2.5.4
                # Filtering of targets: Polymer chain containing fewer than 4
                # resolved residues is filtered out
                polymer_chains[chain_id] = process_polymer_chain(current_chain)
        assert polymer_chains, f"Has no desirable chains for {pdbid}."
        logging.debug(f"{mmcif_path} processed successfully.")
        data = {
            'pdbid': pdbid,
            'structure_method': result.mmcif_object.header['structure_method'],
            'release_date': result.mmcif_object.header['release_date'],
            'resolution': result.mmcif_object.header['resolution'],
            'polymer_chains': polymer_chains,
            'nonpoly_graphs': nonpoly_graphs,
        }
        return data
    except Exception as e:
        logging.error(f"{mmcif_path} processed failed, {e}")
        return {}


def show_one_mmcif(data: Mapping[str, Union[str, dict, list]]) -> None:
    """Show one processed data."""
    print(data.keys())
    print(data['pdbid'])
    print(data['structure_method'])
    print(data['release_date'])
    print("resolution", data['resolution'])
    print('-'*80)
    print("polymer_chains", len(data['polymer_chains']))
    for chain_id, polymer in data['polymer_chains'].items():
        print('-'*80)
        print(polymer.keys())
        print(f"{data['pdbid']}_{chain_id}", end=' ')
        restype = ''.join(polymer['restype'])
        _type = 'protein' if restype.count('p') >= restype.count('n') else 'na'
        print(f"polymer_type={_type} num_residues={len(restype)}")
        is_missing = [np.any(np.isnan(_)) for _ in polymer['center_coord']]
        print(''.join(polymer['seqres']))
        print("".join('-' if _ else c for _, c in zip(is_missing, restype)))
        arr = [f'{_:s}' for _ in polymer['resname'][:10]]
        print(f"resname[:10]          : [{', '.join(arr)}]")
        for i, axis in enumerate('xyz'):
            arr = [f'{_:.3f}' for _ in polymer['center_coord'][:10, i]]
            print(f"center_coord[:10].{axis}   : [{', '.join(arr)}]")
        for i, axis in enumerate('xyz'):
            arr = [f'{_:.3f}' for _ in polymer['allatom_coord'][:10, 0, i]]
            print(f"allatom_coord[:10,0].{axis}: [{', '.join(arr)}]")
    print('-'*80)
    print("nonpoly_graphs", len(data['nonpoly_graphs']))
    for i, graph in enumerate(data['nonpoly_graphs']):
        print('-'*80)
        print(graph.keys())
        print(f"{i}_{data['pdbid']}_{graph['chain_id']}_{graph['name']} "
              f"num_nodes={graph['num_nodes']} "
              f"charge={graph['pdbx_formal_charge']} ",
              end='')
        for k in ['node_coord', 'node_feat', 'edge_index', 'edge_feat']:
            print(f"{k}={graph[k].shape}", end=' ')
        print()
        arr = [f'{_+1:<2}' for _ in graph['node_feat'][:, 0]]
        print("".join(arr))
        arr = [f'{NUM2SYM[_]:<2}' for _ in graph['node_feat'][:, 0]]
        print("".join(arr))
        arr = [f'{_:<2}' for _ in graph['charges']]
        print("".join(arr))
        for i, axis in enumerate('xyz'):
            arr = [f'{_:.3f}' for _ in graph['coords'][:10, i]]
            print(f"coords[:10].{axis}    : [{', '.join(arr)}]")
        for i, axis in enumerate('xyz'):
            arr = [f'{_:.3f}' for _ in graph['node_coord'][:10, i]]
            print(f"node_coord[:10].{axis}: [{', '.join(arr)}]")
        print(f"node_feat[:10,:3]:", graph['node_feat'][:10, :3].tolist())
        print(f"edge_index[:,:10]:", graph['edge_index'][:, :10].tolist())
        print(f"edge_feat[:10,:] :", graph['edge_feat'][:10, :].tolist())


def show_lmdb(lmdbdir: Path):
    with lmdb.open(str(lmdbdir), readonly=True).begin(write=False) as txn:
        metavalue = txn.get('__metadata__'.encode())
        assert metavalue, f"'__metadata__' not found in {lmdbdir}."

        metadata = bstr2obj(metavalue)

        assert 'keys' in metadata, (
            f"'keys' not in metadata for {lmdbdir}.")
        assert 'structure_methods' in metadata, (
            f"'structure_methods' not in metadata for {lmdbdir}.")
        assert 'release_dates' in metadata, (
            f"'release_dates' not in metadata for {lmdbdir}.")
        assert 'resolutions' in metadata, (
            f"'resolutions' not in metadata for {lmdbdir}.")
        assert 'comment' in metadata, (
            f"'comment' not in metadata for {lmdbdir}.")

        print('-'*80)
        print(metadata['comment'], end='')
        for k, v in metadata.items():
            k != 'comment' and print(k, len(v))
        print(f"{len(metadata['keys'])} samples in {lmdbdir}" )
        print(f"metadata['keys'][:10]={metadata['keys'][:10]}")


@cli.command()
@click.option("--mmcif-path",
              type=click.Path(exists=True),
              required=True,
              help="Input path of one mmCIF file rsync from RCSB.")
@click.option("--chem-comp-path",
              type=click.Path(exists=True),
              required=True,
              help="Input mmCIF file of all chemical components.")
def process_one(mmcif_path: str, chem_comp_path: str) -> None:
    """Process one mmCIF file and print the result."""
    mmcif_path = Path(mmcif_path).resolve()
    chem_comp_path = Path(chem_comp_path).resolve()
    print(mmcif_path)
    print(chem_comp_path)
    data = process_one_mmcif(str(mmcif_path), str(chem_comp_path))
    data and show_one_mmcif(data)


@cli.command()
@click.option("--mmcif-dir",
              type=click.Path(exists=True),
              required=True,
              help="Input directory of mmCIF files rsync from RCSB.")
@click.option("--chem-comp-path",
              type=click.Path(exists=True),
              required=True,
              help="Input mmCIF file of all chemical components.")
@click.option("--output-lmdb",
              type=click.Path(exists=False),
              default="output.lmdb",
              help="Output lmdb file.")
@click.option("--num-workers",
              type=int,
              default=-1,
              help="Number of workers.")
@click.option("--data-comment",
              type=str,
              default="PDB snapshot from https://snapshots.pdbj.org/20240101/.",
              help="Comments for output.")
def process(mmcif_dir: str,
            chem_comp_path: str,
            output_lmdb: str,
            num_workers: int,
            data_comment: str,
            ) -> None:
    """Process mmCIF files from directory and save to lmdb."""
    mmcif_dir = Path(mmcif_dir).resolve()
    mmcif_paths = [_ for _ in Path(mmcif_dir).rglob("*.cif.gz")]
    assert mmcif_paths and all(11==len(_.name) for _ in mmcif_paths), (
        f"PDBID should be 4 characters long in {mmcif_dir}.")
    logging.info(f"Processing {len(mmcif_paths)} structures in {mmcif_dir}.")

    chem_comp_path = Path(chem_comp_path).resolve()
    logging.info(f"Chemical components information is in {chem_comp_path}")

    output_lmdb = Path(output_lmdb).resolve()
    assert not output_lmdb.exists(), f"ERROR: {output_lmdb} exists. Stop."
    logging.info(f"Will save processed data to {output_lmdb}")

    env = lmdb.open(str(output_lmdb), map_size=1024**4) # 1TB max size

    metadata = {
        'keys': [],
        'num_polymers': [],
        'num_nonpolys': [],
        'structure_methods': [],
        'release_dates': [],
        'resolutions': [],
        'comment': (
            f'Created time: {datetime.now()}\n'
            f'Input mmCIF: {mmcif_dir}\n'
            f'Chemical components: {chem_comp_path}\n'
            f'Output lmdb: {output_lmdb}\n'
            f'Number of workers: {num_workers}\n'
            f'Comments: {data_comment}\n'
            ),
        }

    pbar = tqdm(total=len(mmcif_paths)//10000+1, desc='Processing chunks (10k)')
    for path_chunk in chunks(mmcif_paths, 10000):
        result_chunk = Parallel(n_jobs=num_workers)(
            delayed(process_one_mmcif)(str(p), str(chem_comp_path))
            for p in tqdm(path_chunk)
            )
        with env.begin(write=True) as txn:
            for data in result_chunk:
                if not data:
                    # skip empty data
                    continue
                txn.put(data['pdbid'].encode(), obj2bstr(data))
                metadata['keys'].append(data['pdbid'])
                metadata['num_polymers'].append(len(data['polymer_chains']))
                metadata['num_nonpolys'].append(len(data['nonpoly_graphs']))
                metadata['structure_methods'].append(data['structure_method'])
                metadata['release_dates'].append(data['release_date'])
                metadata['resolutions'].append(data['resolution'])
        pbar.update(1)
    pbar.close()

    with env.begin(write=True) as txn:
        txn.put('__metadata__'.encode(), obj2bstr(metadata))

    env.close()

    show_lmdb(output_lmdb)


@cli.command()
@click.option("--chem-comp-path",
              type=click.Path(exists=True),
              required=True,
              help="Input mmCIF file of all chemical components.")
@click.option("--sdf-dir",
              type=click.Path(exists=True),
              required=True,
              help="Input directory of SDF files.")
def check_chem_comp(chem_comp_path: str, sdf_dir: str) -> None:
    chem_comp_path = Path(chem_comp_path).resolve()
    logging.info(f"Chemical components information is in {chem_comp_path}")

    sdf_dir = Path(sdf_dir).resolve()
    logging.info(f"The ideal SDF files in {sdf_dir}")

    chem_comp_path = Path(chem_comp_path).resolve()
    chem_comp_full_string = parse_mmcif_string(str(chem_comp_path))
    assert chem_comp_full_string.startswith('data_'), (
        f"Failed to read chem_comp from {chem_comp_path}.")

    chem_comp_strings = split_chem_comp_full_string(chem_comp_full_string)
    assert chem_comp_strings, f"Failed to split {chem_comp_path}."
    logging.info(f"{len(chem_comp_strings)} chem_comp in {chem_comp_path}.")

    def _check_one(name: str) -> int:
        try:
            assert name not in (EXCLUS | GLYCAN), f"{name} in exclusion list."

            sdf_path = sdf_dir / f"{name}_ideal.sdf"
            assert sdf_path.exists(), f"SDF file does not exist for {name}."
            with open(sdf_path, 'r') as fp:
                lines = fp.readlines()
            idx = lines.index('> <OPENEYE_ISO_SMILES>\n')
            smiles = lines[idx+1].strip()
            assert smiles, f"No SMILES in sdf file {name}."

            mol1 = Chem.MolFromSmiles(smiles)
            assert mol1, f"failed to create molecule {name} from SMILES."
            mol1 = Chem.RemoveHs(mol1)
            can1 = Chem.MolToSmiles(mol1, isomericSmiles=False)
            mol1 = Chem.MolFromSmiles(can1)
            can1 = Chem.MolToSmiles(mol1, isomericSmiles=False)
            # iso1 = Chem.MolToSmiles(mol1, isomericSmiles=True)
            # print(can1)

            mol2 = chemcomp2graph(chem_comp_strings[name])['rdkitmol']
            assert mol2, f"failed to create RDKit molecule {name}."
            mol2 = Chem.RemoveHs(mol2)
            can2 = Chem.MolToSmiles(mol2, isomericSmiles=False)
            mol2 = Chem.MolFromSmiles(can2)
            can2 = Chem.MolToSmiles(mol2, isomericSmiles=False)
            # iso2 = Chem.MolToSmiles(mol2, isomericSmiles=True)
            # print(can2)

            return 1 if can1 == can2 else 0
        except Exception as e:
            logging.error(f"Check {name} failed, {e}")
            return -1

    logging.info(f"Checking {len(chem_comp_strings)} chem_comp one by one ...")
    names = sorted(chem_comp_strings.keys())
    results = [(_check_one(_), _) for _ in tqdm(names)]
    STATUS = {1: 'SUCCESS', 0: 'FAILED', -1: 'ERROR'}
    for r in results:
        print(STATUS[r[0]], r[1])


if __name__ == "__main__":
    cli()
