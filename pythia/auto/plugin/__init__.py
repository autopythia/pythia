from dataclasses import MISSING, dataclass, field, fields, is_dataclass, make_dataclass
from functools import cache
from importlib import import_module
import inspect
import pkgutil


@dataclass
class AutopythiaPlugin:
    @classmethod
    def _resolve_fields(cls) -> list:
        return list(fields(cls))

    @classmethod
    def _resolve_post_init(cls):
        v = cls.__dict__.get("_post_init")
        if not isinstance(v, staticmethod):
            return None
        return v.__func__

    @classmethod
    def _resolve_pre_shutdown(cls):
        v = cls.__dict__.get("_pre_shutdown")
        if not isinstance(v, staticmethod):
            return None
        return v.__func__

    @classmethod
    def _resolve_post_shutdown(cls):
        v = cls.__dict__.get("_post_shutdown")
        if not isinstance(v, staticmethod):
            return None
        return v.__func__

    @classmethod
    def _resolve_extensions(cls) -> list:
        named_exts = []
        for name, v in sorted(cls.__dict__.items()):
            if name.startswith("_"):
                continue
            if not isinstance(v, staticmethod):
                continue
            named_exts.append((name, v.__func__))
        return named_exts


def _clone_field(item):
    kwargs = {
        "init": item.init,
        "repr": item.repr,
        "hash": item.hash,
        "compare": item.compare,
        "metadata": item.metadata,
        "kw_only": item.kw_only,
    }
    if item.default is not MISSING:
        kwargs["default"] = item.default
    if item.default_factory is not MISSING:
        kwargs["default_factory"] = item.default_factory
    return field(**kwargs)


@cache
def resolve_plugin_types() -> tuple[type[AutopythiaPlugin], ...]:
    plugin_types = []
    for module_info in sorted(pkgutil.iter_modules(__path__), key=lambda item: item.name):
        module = import_module(f"{__name__}.{module_info.name}")
        for _, value in sorted(vars(module).items()):
            if not inspect.isclass(value):
                continue
            if value is AutopythiaPlugin:
                continue
            if not issubclass(value, AutopythiaPlugin):
                continue
            if value.__module__ != module.__name__:
                continue
            if not is_dataclass(value):
                continue
            plugin_types.append(value)
    return tuple(plugin_types)


def resolve_autopythia_class(base_cls):
    plugin_types = resolve_plugin_types()
    if not plugin_types:
        return base_cls

    base_field_names = set(getattr(base_cls, "__dataclass_fields__", {}))
    resolved_field_names = set()
    resolved_fields = []
    namespace = {
        "__module__": base_cls.__module__,
        "__doc__": base_cls.__doc__,
    }
    post_inits = []
    pre_shutdown_hooks = []
    post_shutdown_hooks = []
    extension_names = []

    for plugin_type in plugin_types:
        for item in plugin_type._resolve_fields():
            if item.name in base_field_names or item.name in resolved_field_names:
                raise ValueError(f"duplicate autopythia plugin field: {item.name}")
            resolved_fields.append((item.name, item.type, _clone_field(item)))
            resolved_field_names.add(item.name)

        post_init = plugin_type._resolve_post_init()
        if post_init is not None:
            post_inits.append(post_init)

        pre_shutdown_hook = plugin_type._resolve_pre_shutdown()
        if pre_shutdown_hook is not None:
            pre_shutdown_hooks.append(pre_shutdown_hook)

        post_shutdown_hook = plugin_type._resolve_post_shutdown()
        if post_shutdown_hook is not None:
            post_shutdown_hooks.append(post_shutdown_hook)

        for name, fun in plugin_type._resolve_extensions():
            if hasattr(base_cls, name) or name in namespace:
                raise ValueError(f"duplicate autopythia plugin extension: {name}")
            namespace[name] = fun
            extension_names.append(name)

    base_post_init = getattr(base_cls, "__post_init__", None)
    base_shutdown = getattr(base_cls, "shutdown", None)

    def __post_init__(self):
        if base_post_init is not None:
            base_post_init(self)
        for post_init in post_inits:
            post_init(self)

    def shutdown(self):
        for pre_shutdown_hook in pre_shutdown_hooks:
            pre_shutdown_hook(self)
        if base_shutdown is not None:
            base_shutdown(self)
        for post_shutdown_hook in post_shutdown_hooks:
            post_shutdown_hook(self)

    namespace["__post_init__"] = __post_init__
    namespace["shutdown"] = shutdown
    namespace["_autopythia_plugin_types"] = plugin_types
    namespace["_autopythia_plugin_extensions"] = tuple(extension_names)

    resolved_cls = make_dataclass(
        base_cls.__name__,
        resolved_fields,
        bases=(base_cls,),
        namespace=namespace,
    )
    resolved_cls.__module__ = base_cls.__module__
    resolved_cls.__qualname__ = base_cls.__qualname__
    return resolved_cls


__all__ = [
    "AutopythiaPlugin",
    "resolve_autopythia_class",
    "resolve_plugin_types",
]
