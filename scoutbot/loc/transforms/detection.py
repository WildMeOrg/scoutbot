# -*- coding: utf-8 -*-
#
#   Copyright EAVISE
#

# from enum import Enum

from scoutbot.loc.transforms import box as b

__all__ = ['Detection', 'ParserType', 'Parser']


class Detection(b.Box):
    """This is a generic detection class that provides some base functionality all detections need.
    It builds upon :class:`~brambox.boxes.box.Box`.

    Attributes:
        confidence (Number): confidence score between 0-1 for that detection; Default **0.0**
    """

    def __init__(self):
        """x_top_left,y_top_left,width,height are in pixel coordinates"""
        super(Detection, self).__init__()
        self.confidence = 0.0  # Confidence score between 0-1

    @classmethod
    def create(cls, obj=None):
        """Create a detection from a string or other box object.

        Args:
            obj (Box or string, optional): Bounding box object to copy attributes from or string to deserialize

        Note:
            The obj can be both an :class:`~brambox.boxes.annotations.Annotation` or a :class:`~brambox.boxes.detections.Detection`.
            For Detections the confidence score is copied over, for Annotations it is set to 1.
        """
        from scoutbot.loc.transforms import annotation as anno

        instance = super(Detection, cls).create(obj)

        if obj is None:
            return instance

        if isinstance(obj, Detection):
            instance.confidence = obj.confidence
        elif isinstance(obj, anno.Annotation):
            instance.confidence = 1.0

        return instance

    def __repr__(self):
        """Unambiguous representation"""
        string = f'{self.__class__.__name__} ' + '{'
        string += f'class_label = {self.class_label}, '
        string += f'object_id = {self.object_id}, '
        string += f'x = {self.x_top_left}, '
        string += f'y = {self.y_top_left}, '
        string += f'w = {self.width}, '
        string += f'h = {self.height}, '
        string += f'confidence = {self.confidence}'
        return string + '}'

    def __str__(self):
        """Pretty print"""
        string = 'Detection {'
        string += f'\'{self.class_label}\'{"" if self.object_id is None else " "+str(self.object_id)}, '
        string += f'[{int(self.x_top_left)}, {int(self.y_top_left)}, {int(self.width)}, {int(self.height)}]'
        string += f', {round(self.confidence*100, 2)}%'
        return string + '}'

    def serialize(self, return_dict=False):
        import json

        serialize_list = [
            self.class_label,
            self.object_id,
            self.x_top_left,
            self.y_top_left,
            self.width,
            self.height,
            self.confidence,
        ]
        if return_dict:
            return serialize_list
        else:
            serialize_str = json.dumps(serialize_list)
            return serialize_str

    def deserialize(self, serialize_str, input_dict=False):
        import json

        if input_dict:
            assert isinstance(serialize_str, dict)
            serialize_list = serialize_str
        else:
            serialize_list = json.loads(serialize_str)
        self.class_label = serialize_list[0]
        self.object_id = serialize_list[1]
        self.x_top_left = serialize_list[2]
        self.y_top_left = serialize_list[3]
        self.width = serialize_list[4]
        self.height = serialize_list[5]
        self.confidence = serialize_list[6]
        return True


ParserType = b.ParserType


class Parser(b.Parser):
    """Generic parser class"""

    box_type = Detection  # Derived classes should set the correct box_type
