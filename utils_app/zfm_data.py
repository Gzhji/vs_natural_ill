import json
from pathlib import Path
from collections import OrderedDict

def load_floor_data(data_path, house_id, floor_id=None, pano_id=None):
    json_file_name = Path(data_path) / house_id / 'zfm_data.json'
    with open(json_file_name) as json_file:
        house_data = json.load(json_file)
    if pano_id is not None:
        floor_id = None
        for floor_id, floor_data in house_data.items():
            if pano_id in floor_data.keys():
                break
        pano_data = house_data[floor_id][pano_id]
        assert(pano_data['is_localized'])
    assert(floor_id is not None)
    floor_data = house_data[floor_id]
    floor_data = OrderedDict(sorted(floor_data.items(), key=lambda t: t[0]))
    print(f'Loading {data_path} {house_id} {floor_id} {pano_id}')
    return floor_id, floor_data
