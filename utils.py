import os

class struct:
    def __str__ (self):
        return "struct{\n    "+"\n   ".join(["{} : {}".format(key, vars(self)[key]) for key in vars(self).keys()]) + "\n}"

def create_dir(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

