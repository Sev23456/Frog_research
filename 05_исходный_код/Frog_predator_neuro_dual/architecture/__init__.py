"""Architecture module - exports main components"""

from Frog_predator_neuro_dual.architecture.visual_system import RetinalProcessing
from Frog_predator_neuro_dual.architecture.tectum import Tectum
from Frog_predator_neuro_dual.architecture.motor_hierarchy import MotorHierarchy
from Frog_predator_neuro_dual.architecture.spatial_memory import SpatialMemory
from Frog_predator_neuro_dual.architecture.basal_ganglia import BasalGanglia

__all__ = [
    'RetinalProcessing',
    'Tectum',
    'MotorHierarchy',
    'SpatialMemory',
    'BasalGanglia',
]

