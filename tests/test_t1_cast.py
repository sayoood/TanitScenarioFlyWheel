from tsf.closedloop.neural import vehicle_cast

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
    last = {"front": {"size": SIZE, "objects": [_o("truck", [0, 0, 1500, 1084], 2.0),
                                                _o("lead", [1400, 450, 1500, 520], 28.0),
                                                _o("far", [100, 500, 140, 520], 60.0)]},
            "rear": {"size": SIZE, "objects": [_o("suv", [0, 0, 300, 200], 5.0), _o("far", [5, 5, 40, 9], 60.0)]}}
    cast = vehicle_cast(TSIR, [first, last], ["front", "left", "rear"])
    assert cast["front"] == ("Exactly three vehicles are in view. A white truck is in the middle, close. "
                             "A blue van is on the right. A red car is on the left, far away.")
    assert cast["left"] == "The road in view is empty."
    assert cast["rear"] == "Exactly one vehicle is in view. An SUV is on the left, close."    # 'far': 4 px tall


def test_no_projection_no_cast():
    assert vehicle_cast(TSIR, [None, None], ["front"]) == {}
