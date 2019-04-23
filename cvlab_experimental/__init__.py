from cvlab.view.config import ConfigWrapper, ELEMENTS_SECTION, EXPERIMENTAL_ELEMENTS
if ConfigWrapper.get_settings().get_with_default(ELEMENTS_SECTION, EXPERIMENTAL_ELEMENTS) == "True":
    from cvlab.diagram.elements import load_auto, ignored_modules
    ignored_modules += ["sample"]
    load_auto(__file__)