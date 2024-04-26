# -*- coding: utf-8 -*-
import json
import os

import mysql.connector
from tqdm import tqdm


def connect_db():
    cnx = mysql.connector.connect(
        # host="your_host",
        user="FAREAST.renqianluo",
        password=os.getenv("DB_PASSWORD"),
        database="qmdb_v1_5_102021",
    )
    cursor = cnx.cursor()
    return cursor, cnx


# column_names = ["id", "entry_id", "reference_id", "label", "prototype_id", "measured", "composition_id", "natoms", "nsites", "ntypes",
#                 "x1", "x2", "x3", "y1", "y2", "y3", "z1", "z2", "z3", "sxx", "syy", "szz", "sxy", "syz", "szx", "spacegroup_id",
#                 "energy", "energy_pa", "magmom", "magmom_pa", "delta_e", "meta_stability", "fit_id", "volume", "volume_pa", "coords",
#                 "id", "structure_id", "site_id", "element_id", "ox", "x", "y", "z",
#                 "fx", "fy", "fz", "magmom", "charge", "volume", "occupancy", "wyckoff_id"]


def format_row(row):
    return {
        "source": "qmdb",
        "id": row[0],
        "formula": row[1],
        "natoms": row[2],
        "nsites": row[3],
        "ntypes": row[4],
        "lattice": [
            [row[5], row[6], row[7]],
            [row[8], row[9], row[10]],
            [row[11], row[12], row[13]],
        ],
        "space_group": {"no": row[14]},
        "energy": row[15],
        "energy_pa": row[16],
        "sites": [],
    }


def dump_db(cursor, cnx):
    query = (
        "SELECT structures.id, structures.composition_id, structures.natoms, structures.nsites, structures.ntypes, "
        "structures.x1, structures.x2, structures.x3, structures.y1, structures.y2, structures.y3, structures.z1, structures.z2, structures.z3, "
        "structures.spacegroup_id, structures.energy, structures.energy_pa, "
        "atoms.element_id, atoms.x, atoms.y, atoms.z "
        "from structures INNER JOIN atoms on structures.id=atoms.structure_id;"
    )
    print(query)
    cursor.execute(query)
    rows = cursor.fetchall()
    # column_names = [i[0] for i in cursor.description]

    res = {}
    with open("/hai1/SFM/threedimargen/data/materials_data/qmdb.jsonl", "w") as f:
        for row in tqdm(rows):
            if not res or row[0] != res.get("id"):
                if res:
                    f.write(json.dumps(res) + "\n")
                res = format_row(row)
            res["sites"].append(
                {
                    "element": row[-4],
                    "fractional_coordinates": [row[-3], row[-2], row[-1]],
                }
            )
        if res:
            f.write(json.dumps(res) + "\n")

    cursor.close()
    cnx.close()


if __name__ == "__main__":
    cursor, cnx = connect_db()
    dump_db(cursor, cnx)
