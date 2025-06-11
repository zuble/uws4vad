class Registry:
    def __init__(self):
        self._registry = {
            "network": {},
            "fm": {},
            "cls": {},
            "loss": {}
        }
        print("Registry initialized")

    def register_network(self, name):
        return self._register("network", name)

    def register_fm(self, name):
        return self._register("fm", name)

    def register_loss(self, name):
        return self._register("loss", name)

    def _register(self, component_type, name):
        def decorator(cls):
            if name in self._registry[component_type]:
                raise ValueError(f"{component_type}.{name} already registered.")
            self._registry[component_type][name] = cls
            return cls
        return decorator
    
    def get(self, component_type, name):
        if component_type not in self._registry:
            raise KeyError(f"Component type '{component_type}' not found.")
        if name not in self._registry[component_type]:
            raise KeyError(f"Component '{name}' not found for type '{component_type}'.")
        return self._registry[component_type][name]
    
# Create a single global registry instance
registry = Registry()
