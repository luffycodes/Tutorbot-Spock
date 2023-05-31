import json

from convert_convs_to_vicuna_v3 import produce_conv_json
from convert_problems_to_vicuna import produce_prob_json

arr1 = produce_prob_json()
arr2 = produce_conv_json()

for c in arr2:
    arr1.append(c)

with open('spock_conv_gpt4_w_index_v3.json', "w") as f:
    json.dump(arr1, f, indent=4)
