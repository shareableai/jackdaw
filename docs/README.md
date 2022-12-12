# Jackdaw Documentation

## General Topics
### [Saving & Loading Models](save.md)
How saving models works in Jackdaw, as well as how to use tracing to check which items are saved on each model.

### [Discovering Models](discover.md)
How to retrieve models programmatically, as well as searching for models using their attributes, metrics, etc.

## Customising Jackdaw to your Models
### [Detectors](detection.md)
Leading on from how saving works - creating classes that allow Jackdaw to identify models it hasn't seen before.

### [Serializers](serializers.md)
Serializers are how Jackdaw saves models to disk, and recreates them in Python. This topic goes into more depth in 
how serializers work within Jackdaw, and how to create serializers for new models or model items. 

### [Access Interface](access_interface.md)
More complex model structures can require an Access Interface, a way of retrieving attributes of a class. This topic
goes into detail on how classes like Torch's nn.ModuleList and nn.Sequential are processed in Jackdaw, and how to 
process similar model classes.