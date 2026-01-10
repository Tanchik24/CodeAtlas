from enum import Enum


class NodeLabel(str, Enum):
    PROJECT = "Project"
    PACKAGE = "Package"
    FOLDER = "Folder"
    MODULE = "Module"
    CLASS = "Class"
    FUNCTION = "Function"
    METHOD = "Method"