import cv2
import numpy as np

from tsf.closedloop.neural import seen_colours, settle_looks, vehicle_cast

TSIR = {"actors": [{"id": "ego", "role": "ego", "category": "car", "color": "white"},
                   {"id": "lead", "category": "van", "color": "blue"},
                   {"id": "truck", "category": "truck", "color": "white"},
                   {"id": "suv", "category": "suv"},
                   {"id": "far", "category": "car", "color": "red"}]}
SIZE = [1924, 1084]


def _o(oid, box, depth):
    return {"id": oid, "type": "Car", "box": box, "depth": depth}


def test_each_camera_names_the_vehicles_the_map_puts_in_it_nearest_first_and_where():
    first = {"front": {"size": SIZE, "objects": [_o("lead", [1400, 450, 1500, 520], 30.0),
                                                 _o("far", [10, 500, 14, 503], 140.0)]},
             "left": {"size": SIZE, "objects": []}}
    last = {"front": {"size": SIZE, "objects": [_o("truck", [300, 0, 1500, 1084], 2.0),
                                                _o("lead", [1400, 450, 1500, 520], 28.0),
                                                _o("far", [100, 500, 140, 520], 60.0)]},
            "rear": {"size": SIZE, "objects": [_o("suv", [0, 0, 300, 200], 5.0), _o("far", [5, 5, 40, 9], 60.0)]}}
    cast = vehicle_cast(TSIR, [first, last], ["front", "left", "rear"])
    assert cast["front"] == ("Exactly three vehicles are in view. A white truck is in the middle, close. "
                             "A blue van is on the right. A red car is on the left, far away.")
    assert cast["left"] == "The road in view is empty."
    assert cast["rear"] == "Exactly one vehicle is in view. An SUV is on the left, close."    # 'far': 4 px tall


def test_a_vehicle_hidden_behind_a_nearer_one_is_not_named():
    """The yellow bus behind the van beside the ego: named, its colour went onto the truck in view."""
    proj = {"front": {"size": SIZE, "objects": [_o("lead", [1300, 150, 1924, 1084], 1.0),
                                                _o("far", [1500, 450, 1900, 600], 12.0),       # inside the van's box
                                                _o("truck", [700, 450, 1400, 620], 25.0)]}}    # mostly in view
    assert vehicle_cast(TSIR, [proj, None], ["front"])["front"] == (
        "Exactly two vehicles are in view. A blue van is on the right, close. A white truck is in the middle.")
    clear = {"front": {"size": SIZE, "objects": [_o("far", [1500, 450, 1900, 600], 12.0)]}}
    assert "A red car is on the right." in vehicle_cast(TSIR, [proj, clear], ["front"])["front"]  # seen in one


def test_no_projection_no_cast():
    assert vehicle_cast(TSIR, [None, None], ["front"]) == {}


def test_a_redraw_names_a_vehicle_by_the_look_it_was_given():
    proj = {"front": {"size": SIZE, "objects": [_o("lead", [1400, 450, 1500, 520], 30.0)]}}
    assert vehicle_cast(TSIR, [proj, None], ["front"], looks={"lead": "black"})["front"] == (
        "Exactly one vehicle is in view. A black van is on the right.")


def _paint(path, boxes):
    """A 720p frame: sky over a warm grey road, each (proj box, BGR) filled in."""
    img = np.full((720, 1280, 3), (200, 180, 160), np.uint8)
    img[360:] = (110, 120, 130)
    s = 1280 / SIZE[0]
    for (u0, v0, u1, v1), bgr in boxes:
        img[round(v0 * s):round(v1 * s), round(u0 * s):round(u1 * s)] = bgr
    cv2.imwrite(str(path), img)
    return str(path)


def test_seen_colours_tells_coloured_vehicles_from_black_and_white_ones(tmp_path):
    """The blue van and the yellow bus (its dark windscreen filling most of the box) show their colour; the black car
    and the white truck under the road's warm cast do not; a far car (40 px) and one hidden behind the black car are
    not judged."""
    objs = [_o("van", [200, 600, 500, 900], 10.0), _o("car", [600, 600, 900, 900], 10.0),
            _o("truck", [1000, 500, 1300, 900], 12.0), _o("bus", [1400, 500, 1800, 900], 15.0),
            _o("far", [100, 550, 160, 580], 60.0), _o("hidden", [650, 650, 850, 850], 20.0)]
    front = _paint(tmp_path / "front.png", [([200, 600, 500, 900], (120, 60, 30)), ([600, 600, 900, 900], (20, 20, 22)),
                                            ([1000, 500, 1300, 900], (215, 228, 235)),
                                            ([1400, 500, 1800, 900], (40, 180, 220)), ([1400, 560, 1800, 760], (30, 30, 30)),
                                            ([100, 550, 160, 580], (40, 40, 200))])
    proj = {"front": {"size": SIZE, "objects": objs}}
    assert seen_colours({"front": front}, proj) == {"van": None, "car": "black", "truck": "white", "bus": None}
    # the camera where a vehicle's box is widest decides: the van close beside, black there
    side = _paint(tmp_path / "side.png", [([100, 300, 1500, 1000], (25, 22, 20))])
    proj["side"] = {"size": SIZE, "objects": [_o("van", [100, 300, 1500, 1000], 3.0)]}
    assert seen_colours({"front": front, "side": side}, proj)["van"] == "black"
    assert seen_colours({"front": front}, None) == {}


def test_a_look_changes_only_when_two_verdicts_in_a_row_agree():
    seen, history, looks = {}, {}, []
    for v in ["black", None, "black", "dark grey", None, "black", None, None]:
        settle_looks(seen, history, {"van": v})
        looks.append(seen.get("van", "-"))
    assert looks == ["-", "-", "-", "dark grey", "dark grey", "dark grey", "dark grey", None]
