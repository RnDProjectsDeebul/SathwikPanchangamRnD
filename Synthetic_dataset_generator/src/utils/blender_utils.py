# import modules
import bpy
import math
import random
import time
from mathutils import Euler, Color
from pathlib import Path

# We need basic blender utilities in this file.

# Class for blender utilities.
class Blender():
    def __init__(self) -> None:
        pass
    
    # function for random rotation
    def set_random_rotation(self,obj_name):
        """
        Applies a random rotation to the given object.

        Parameters:
            obj_name: str
        """
        obj_to_change = obj_name
        random_rotat_values = [random.random()*2*math.pi,random.random()*2*math.pi,random.random()*2*math.pi]
        obj_to_change.rotation_euler = Euler(random_rotat_values,'XYZ')

    # function for changing the materials of an object
    def change_material(self,obj_name):
        """
        Changes the materials for the given object randomly using the existing materails.

        Parameters:
            obj_name: str
        """
    
        # Set the object to active object.
        object = obj_name
        bpy.context.view_layer.objects.active = None
        bpy.context.view_layer.objects.active = object
    
        # Get the list of materials.
        materials_list = bpy.data.materials.keys()
        materials_list.remove('Dots Stroke')
    

        # Enable edit mode to assign the materials. Assuming object is in object mode.
        bpy.ops.object.editmode_toggle()
    
        # Set the material to apply by providing the index
        bpy.context.object.active_material_index = random.choice(range(0,len(materials_list)))
    
        # Assign the material
        bpy.ops.object.material_slot_assign()
    
        # Change back to object mode.
        bpy.ops.object.editmode_toggle()
    
        # print(materials_list)
    
    # function to change random colors of an object
    def change_color(material_name):
        """
        Applies Materials randomly from the defined materials.

        Changes specially the Principled BSDf color values for the given material.
    
        Parameters:
            material_name: str
        """
        material_to_change = bpy.data.materials[material_name]
    
        color = Color()
        hue = random.random()  # Random hue between 0 and 1
        color.hsv = (hue,1,1)
    
        rgba = [color.r, color.g, color.b, 1]
    
        material_to_change.node_tree.nodes['Principled BSDF'].inputs[0].default_value = rgba
    
    # function for random lighting conditions
    def set_random_lighting(self,light_source_name,min_value,max_value):
        """
        Applies random light intensities to the scene.

        Parameters:
            light_source_name:str
            min_value: float
            max_value: float
        """

        bpy.data.lights[str(light_source_name)].energy = random.uniform(min_value,max_value)

    # function to adjust the camera position and light position in the scene.
    # this can be based on the transformation matrix. you can wite seperate function
    # for the transformation matrix.
    def transform_camera_light_to_objects(self):
        pass

    # function to avoid interference of objects with other objects,
    # you can use minimum distance.
    def set_minimum_distnce(self):
        pass


    # Function to set render parameters
    # ex: cycles, denoise, no.of samples, etc
    def set_render_parameters(self,image_resolution):
        pass


# Future work

# funciton to delete every thing in the scene 
# and add the background, camera and light source with their respective locations.
# Also set all the objects to the center.
def setup_scene():
    pass

# funciton to add all objects into the scene preferablly at center.

# Function for random data augmentations mirror horizontal flip or vertical flip etc. but not necessary low priority.
# if this function is enabled the user should get the dataset with some random augmentations


# Main assumptions in context of uncertainty generator.

