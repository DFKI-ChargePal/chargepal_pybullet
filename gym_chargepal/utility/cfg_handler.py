# global
import copy

# mypy
from typing import Any, Dict, List


class ConfigHandler:

    def update(self, **kwargs: Any) -> None:
        """ Update configurations values """
        for attr_name, new_attr_value in kwargs.items():
            if hasattr(self, attr_name):
                setattr(self, attr_name, copy.deepcopy(new_attr_value))

    def is_valid(self) -> bool:
        """ Check if all attributes are occupied with values """
        retval = True
        for attr in dir(self):
            if not attr.startswith('__') and not callable(getattr(self, attr)):
                if getattr(self, attr) is None:
                    retval = False
        return retval

    def post_init_check(self) -> None:
        """ Check if object is fully initialized.
        The check pass when none of the class attributes has the value 'None'
        """
        none_flag = False
        none_list: List[str] = []
        for attr in dir(self):
            if not attr.startswith('__') and not callable(getattr(self, attr)):
                if getattr(self, attr) is None:
                    none_list.append(attr)
                    none_flag = True
        if none_flag:
            msg_attachment = f"None attributes are: '{none_list}'"
            raise AttributeError(f"Configuration object is not fully initialized. {msg_attachment}")


def search(config: Dict[str, Any], name: str) -> Dict[str, Any]:
    sub_cfg: Dict[str, Any] = {} if name not in config else copy.deepcopy(config[name])
    return sub_cfg
