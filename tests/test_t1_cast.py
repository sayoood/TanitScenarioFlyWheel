from tsf.closedloop.neural import vehicle_cast

TSIR = {"actors": [{"id": "ego", "role": "ego", "category": "car", "color": "white"},
                   {"id": "lead", "category": "van", "color": "blue"},
                   {"id": "truck", "category": "truck", "color": "white"},
                   {"id": "suv", "category": "suv"},
                   {"id": "far", "category": "car", "color": "red"}]}


def _o(oid, box, depth):
    return {"id": oid, "type": "Car", "box": box, "depth": depth}


def test_each_camera_names_the_vehicles_the_map_puts_in_it_nearest_first():
    first = {"front": {"objects": [_o("lead", [900, 450, 1000, 520], 30.0), _o("far", [10, 500, 14, 503], 140.0)]},
             "left": {"objects": []}}
    last = {"front": {"objects": [_o("truck", [0, 0, 1500, 1084], 2.0), _o("lead", [900, 450, 1000, 520], 28.0)]},
            "rear": {"objects": [_o("suv", [0, 0, 300, 200], 5.0), _o("far", [5, 5, 40, 9], 60.0)]}}
    cast = vehicle_cast(TSIR, [first, last], ["front", "left", "rear"])
    assert cast["front"] == "The only vehicles in view are a white truck and a blue van."   # 'far': beyond 120 m
    assert cast["left"] == "The road in view is empty."
    assert cast["rear"] == "The only vehicle in view is an SUV."                            # 'far': 4 px tall


def test_no_projection_no_cast():
    assert vehicle_cast(TSIR, [None, None], ["front"]) == {}
