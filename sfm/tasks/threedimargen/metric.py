# -*- coding: utf-8 -*-
from multiprocessing import Pool, TimeoutError

from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core.structure import Structure

LTOL = 0.3
STOL = 0.5
ANGLE_TOL = 10


def handler(signum, frame):
    raise Exception("Function call timed out")


def get_pmg_structure(sites, lattice, coordinates):
    structure = Structure(lattice, sites, coordinates)
    return structure


def get_rms_dist(structure1, structure2):
    matcher = StructureMatcher(ltol=LTOL, stol=STOL, angle_tol=ANGLE_TOL)
    structure1 = get_pmg_structure(
        structure1["sites"], structure1["lattice"], structure1["coordinates"]
    )
    structure2 = get_pmg_structure(
        structure2["sites"], structure2["lattice"], structure2["coordinates"]
    )
    rms_dist = matcher.get_rms_dist(structure1, structure2)
    return rms_dist


def if_match(structure1, structure2):
    matcher = StructureMatcher(ltol=LTOL, stol=STOL, angle_tol=ANGLE_TOL)
    structure1 = get_pmg_structure(
        structure1["sites"], structure1["lattice"], structure1["coordinates"]
    )
    structure2 = get_pmg_structure(
        structure2["sites"], structure2["lattice"], structure2["coordinates"]
    )

    with Pool(1) as pool:
        res = pool.apply_async(matcher.fit, (structure1, structure2))
        try:
            if_match = res.get(timeout=2)
        except TimeoutError:
            print(f"Timeout\n{structure1}\n{structure2}\n")
            if_match = False
    return if_match


if __name__ == "__main__":
    import json

    data = json.loads(
        """{"source": "mp_20", "id": 3310, "material_id": "mp-28968", "formula": "Nb6 Te6 As2", "lattice": [[0.0, 0.0, -3.583128], [-5.26535870151421, -9.11986948216934, -1.2896410826211497e-15], [-5.26535989848579, 9.11986948216934, 6.448205413105748e-16]], "sites": [{"element": "Nb", "fractional_coordinates": [0.25, 0.3783120000000001, 0.504061], "cartesian_coordinates": [-4.64601095687789, 1.1468144675133112, -0.8957820000000001]}, {"element": "Nb", "fractional_coordinates": [0.25, 0.49594000000000016, 0.874252], "cartesian_coordinates": [-7.214553416399957, 3.4501560635384463, -0.8957820000000001]}, {"element": "Nb", "fractional_coordinates": [0.75, 0.621688, 0.495939], "cartesian_coordinates": [-5.884707643122111, -1.1468144675133125, -2.6873460000000002]}, {"element": "Nb", "fractional_coordinates": [0.25, 0.12574900000000022, 0.6216890000000002], "cartesian_coordinates": [-3.9355299212864443, 4.522908070987063, -0.8957819999999996]}, {"element": "Nb", "fractional_coordinates": [0.75, 0.5040600000000001, 0.12574800000000003], "cartesian_coordinates": [-3.3161651836000443, -3.450156063538448, -2.6873460000000002]}, {"element": "Nb", "fractional_coordinates": [0.75, 0.874251, 0.37831099999999995], "cartesian_coordinates": [-6.595188678713558, -4.522908070987063, -2.6873460000000007]}, {"element": "Te", "fractional_coordinates": [0.25, 0.2773350000000001, 0.947144], "cartesian_coordinates": [-6.447322291175869, 6.108570657982362, -0.8957819999999996]}, {"element": "Te", "fractional_coordinates": [0.75, 0.7226650000000001, 0.052856000000000014], "cartesian_coordinates": [-4.083396308824132, -6.108570657982365, -2.6873460000000007]}, {"element": "Te", "fractional_coordinates": [0.75, 0.3301910000000001, 0.27733499999999994], "cartesian_coordinates": [-3.1988426424582355, -0.48203982134954426, -2.6873460000000002]}, {"element": "Te", "fractional_coordinates": [0.25, 0.052856000000000236, 0.330191], "cartesian_coordinates": [-2.0168802497681577, 2.529259002837432, -0.8957819999999999]}, {"element": "Te", "fractional_coordinates": [0.75, 0.947144, 0.669809], "cartesian_coordinates": [-8.513838350231843, -2.529259002837433, -2.6873460000000007]}, {"element": "Te", "fractional_coordinates": [0.25, 0.6698090000000001, 0.722665], "cartesian_coordinates": [-7.331875957541765, 0.4820398213495416, -0.8957820000000004]}, {"element": "As", "fractional_coordinates": [0.2500000000000001, 0.6666666666666667, 0.33333333333333326], "cartesian_coordinates": [-5.265359100504737, -3.039956494056448, -0.8957820000000011]}, {"element": "As", "fractional_coordinates": [0.75,0.3333333333333335, 0.6666666666666666], "cartesian_coordinates": [-5.265359499495264, 3.0399564940564447, -2.687346]}], "space_group": {"no": 176, "symbol": "P6_3/m"}, "prediction": {"lattice": [[7.778717517852783, -2.4276840686798096, -0.06909847259521484], [7.002684116363525, 2.5673606395721436, -0.04682976007461548], [-4.069267749786377, 6.982647895812988, 0.054802775382995605]], "coordinates": [[0.0428036630153656, 0.038502898812294004, 0.6273246288299561], [0.3721041202545166, 0.21752350330352782, 0.21070284843444825], [0.5587663650512695, 0.518076229095459, 0.5372518062591553], [0.4920063018798828, 0.5312482833862304, 0.5996757030487061], [0.09280890226364136, 0.29208099842071533, 0.3045450210571289], [0.786902904510498, 0.7409423351287842, 0.45864405632019045], [0.45771188735961915, 0.43240671157836913, 0.4660821437835693], [0.4807030200958252, 0.26134800910949707, 0.4205376625061035], [0.5890142917633057, 0.4775557041168213, 0.5460916519165039], [0.5223749160766602, 0.586772346496582, 0.4493544101715088], [0.28228025436401366, 0.48227567672729493, 0.5478128910064697], [0.6721799850463868, 0.7993371963500977, 0.4943880558013916], [0.5360360145568848, 0.412760591506958, 0.49371795654296874], [0.2225719928741455, 0.10775806903839111, 0.33104169368743896]]}}"""
    )
    sites_list = [site["element"] for site in data["sites"]]
    s_predict = {
        "sites": sites_list,
        "lattice": data["prediction"]["lattice"],
        "coordinates": data["prediction"]["coordinates"],
    }
    s_gt = {
        "sites": sites_list,
        "lattice": data["lattice"],
        "coordinates": [site["fractional_coordinates"] for site in data["sites"]],
    }
    match = if_match(s_predict, s_gt)
    if match:
        rms_dist = get_rms_dist(s_predict, s_gt)
        print(rms_dist)
    else:
        print("not match")
