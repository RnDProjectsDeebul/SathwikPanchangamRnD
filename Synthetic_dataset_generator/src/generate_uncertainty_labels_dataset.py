# import modules
import os
import sys
import bpy


# get the arguments from the command line
argv = sys.argv
argv = argv[argv.index("--") + 1:] # get all the arguments after "--"

print(argv) # prints a list of arguments