"""
Seeding Module

Source detection and seed model generation for HAWC analysis.
Supports multiple algorithms: DRIPS (image-based) and ALPS (iterative).

Usage:
------
from seeding.base import SeedingModule, SeedingOutput
from seeding.image_seeds import DRIPSSeeder
from seeding.alps_seeder import ALPSSeeder
"""

from seeding.base import SeedingModule, SeedingOutput
from seeding.image_seeds import DRIPSSeeder
from seeding.alps_seeder import ALPSSeeder

__all__ = ['SeedingModule', 'SeedingOutput', 'DRIPSSeeder', 'ALPSSeeder']
