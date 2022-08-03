# **Synthetric dataset generator using Blender**

Synthetic dataset generation library in Python for classification, object detection, and uncertainty labels datasets using blender.

## **Requirements:**
* [Blender](https://www.blender.org/)
* Cad models for objects along with their materials and textures.

## **Instructions**
1. Install requirements from [requirements.txt](/Synthetic_dataset_generator/requirements.txt) file
2. Create cad models for the desired objects along with their materials and textures.
### **_Classification dataset generation:_**
* For generating classification samples run:
```
blender --background blend_file.blend --python generate_classification_dataset.py --infile json_file.json
```
* Replace the blend_file.blend with your blender file containing CAD models.
* For the args you can specify a json file with all arguments. provide an example for that in the readme. 
* This can include the path for saving the dataset, number of images to render, image resolution, what kind of augmentations are needed (random rotation, random lighting) you can provide this as a boolean value. ex. random_lighting:True, random_rotation:False,random_background:False etc
#### **_Generated classification examples:_**
![alt tag](/images/addimagehere)

### **_Object detection dataset generation:_**

* For generating object detection samples run:
```
blender --background blend_file.blend --python generate_object_detection_dataset.py --infile json_file.json
```
* Replace the blend_file.blend with your blender file containing CAD models.

#### **_Generated object detection examples_**

![alt tag](/images/addimagehere)

### **_Uncertainty labels dataset generation:_**
* For generating object detection samples run:

```
blender --background blend_file.blend --python generate_uncertainty_labels_dataset.py --infile json_file.json
```
* Replace the blend_file.blend with your blender file containing CAD models.

#### **_Generated uncertainty labels examples_**
![alt tag](/images/addimagehere)

