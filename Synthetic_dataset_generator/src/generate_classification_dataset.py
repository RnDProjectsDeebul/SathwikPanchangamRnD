# import modules
import bpy
import os
import sys
import json
import argparse
from pathlib import Path
import time

# append the working directory so that blender will recognise the custom modules.
sys.path.append('/home/sathwikpanchngam/rnd/github_projects/SathwikPanchangamRnD/Synthetic_dataset_generator/')

#sys.path.append(os.getcwd())

from src.utils.blender_utils import Blender
print('All modules are sucessfully imported')

# # parse the json file from the command line.
# parser = argparse.ArgumentParser()
# parser.add_argument('--infile', nargs=1,help="JSON file to be processed",type=argparse.FileType('r'))
# arguments = parser.parse_args()

# # Loading the JSON file as a dictionary
# parameters_dict = json.load(arguments.infile[0])
# print(parameters_dict)
# print(parameters_dict['Image_resolution'])
# print(type(parameters_dict['Image_resolution']))

# tupled = tuple(parameters_dict['Image_resolution'])
# print(tupled)
# print(type(tupled))
# print(type(int(tupled[0])))
# print(tupled[0]*2)
# print(int(tupled[0])*2)


# initilize custom module class
blender_instance = Blender()


# get object names
obj_names,obj_count = blender_instance.get_obj_names_count()

print('Names : ',obj_names)
print('Num_objects : ',obj_count)

# Set output path for saving the dataset.
output_path = Path('/home/sathwikpanchngam/rnd/Datasets/synthetic_datasets/11_classes_new')


# Set how many images to render per class
obj_renders_per_split = [('train',30)]

total_render_count = sum([obj_count*r[1] for r in obj_renders_per_split])

print('Num_train_images_per_class : ',obj_renders_per_split[0][1])
print('Total images to render : ', total_render_count)


# set all objects to be hidden in rendering
for name in obj_names:
    bpy.context.scene.objects[name].hide_render = True
    # Tracks the starting image index for each object loop 
    start_idx = 0
    start_time = time.time()

# Loop through each split of the dataset.
for split_name , renders_per_object in obj_renders_per_split:
    print(f'Starting split: {split_name} | Total renders: {renders_per_object * obj_count}')
    print('==============================================')
     
    # Loop through the object name
    for obj_name in obj_names:
        print(f'Starting object: {split_name}/{obj_name}')
        print('..................................................')
        
        # Get the next object and make it visible
        obj_to_render = bpy.context.scene.objects[obj_name]
        obj_to_render.hide_render = False
        
        # Loop through all image renders for this object
        for i in range(start_idx, start_idx + renders_per_object):
            # Change the object
            blender_instance.set_random_rotation(obj_to_render)
            blender_instance.set_random_lighting(light_source_name='Sun',min_value=0.5,max_value=10)
#            change_material(bpy.context.scene.objects['Floor'].material_slots[0].material)

            
            # Log status
            print(f'Rendering image {i +1} of {total_render_count}')
            seconds_per_render = (time.time() - start_time) / (i+1)
            seconds_remaining = seconds_per_render * (total_render_count - i -1)
            print(f'Estimated time remaining: {time.strftime("%H:%M:%S", time.gmtime(seconds_remaining))}')
            
            # Update file path and render
            bpy.context.scene.render.filepath = str(output_path / split_name / obj_name / f'{str(i).zfill(6)}.png')
            bpy.ops.render.render(write_still = True)
            
        # Hide the object 
        obj_to_render.hide_render = True
            
        
        # Update the starting index 
        start_idx += renders_per_object
        
        
# Set all objects to be visible in rendering 
for name in obj_names:
    bpy.context.scene.objects[name].hide_render = False




   

#if __name__ =='__main__':
#    main()
## setup the scene in blender for now istead of setting it up from code.

## Get the list of objects present in the scene.
## obj_names = bpy.context.scene.objects.keys()
## print(obj_names)
## obj_names.remove('Camera')
## obj_names.remove('Sun')
## obj_names.remove('Floor')
## obj_count = len(obj_names)
## print("Number of objects in the scene: ", obj_count)

## # specify the number of images required for training, test and validation datasets.
## obj_render_per_split = [('train',2),('test',2),('val',2)]

## # Specify the output path for saving the images
## output_path = parameters_dict['Path']
## print(output_path,type(output_path))