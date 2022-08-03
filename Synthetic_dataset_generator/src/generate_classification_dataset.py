# import modules
import bpy
import os
import sys
import json
import argparse

from utils.blender_utils import Blender

# parse the json file from the command line.
parser = argparse.ArgumentParser()
parser.add_argument('--infile', nargs=1,help="JSON file for blender parameters",type=argparse.FileType('r'))
arguments = parser.parse_args()

# Loading the JSON file as a dictionary
parameters_dict = json.load(arguments.infile[0])
print(parameters_dict)

print(parameters_dict['Image_resolution'])
print(type(parameters_dict['Image_resolution']))

tupled = tuple(parameters_dict['Image_resolution'])
print(tupled)
print(type(tupled))
print(type(int(tupled[0])))
print(tupled[0]*2)
print(int(tupled[0])*2)


# setup the scene in blender for now istead of setting it up from code.

# Get the list of objects present in the scene.
# obj_names = bpy.context.scene.objects.keys()
# print(obj_names)
# obj_names.remove('Camera')
# obj_names.remove('Sun')
# obj_names.remove('Floor')
# obj_count = len(obj_names)
# print("Number of objects in the scene: ", obj_count)

# # specify the number of images required for training, test and validation datasets.
# obj_render_per_split = [('train',2),('test',2),('val',2)]

# # Specify the output path for saving the images
# output_path = parameters_dict['Path']
# print(output_path,type(output_path))