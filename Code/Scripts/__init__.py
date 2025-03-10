
#------------------------------------------------------------------------
#
#    PestoSeis, a numerical laboratory to learn about seismology, written
#    in the Python language.
#    Copyright (C) 2022  Andrea Zunino 
#
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
#------------------------------------------------------------------------

""" 
  Functions to calculate acoustic and elastic wave propagation in 2D.
"""

__all__ = ["solveacoustic2D","solveelastic2D","gaussource","rickersource","animateacousticwaves","animateelasticwaves"]

from .USwaveprop2D import solveacoustic2D

#from .acousticwaveprop2D import solveacoustic2D

from .elasticwaveprop2D import solveelastic2D

from .sourcetimefuncs import gaussource, rickersource

from .animatewaves import animateacousticwaves,animateelasticwaves
