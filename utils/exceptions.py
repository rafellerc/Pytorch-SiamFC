class IncompatibleFolderStructure(Exception):
    pass


class IncompatibleImagenetStructure(IncompatibleFolderStructure):
    def __init__(self, msg=None):
        info = ("\n"
                "The given root directory does not conform with the expected "
                "folder structure. It should have the following structure:\n"
                "Imagenet/\n"
                "├── Annotations\n"
                "│   └── VID\n"
                "│       ├── train\n"
                "│       └── val\n"
                "└── Data\n"
                "    └── VID\n"
                "        ├── train\n"
                "        └── val\n")
        if msg is not None:
            info = info + msg
        super().__init__(info)


class InvalidOption(Exception):
    pass


class IncompleteArgument(Exception):
    pass