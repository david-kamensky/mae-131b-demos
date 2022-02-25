# FEniCS-based lecture demos for MAE 131B
This repository contains several FEniCS scripts to illustrate various results in linear elasticity.  It also contains a MATLAB/[Octave](https://www.gnu.org/software/octave/index) script implementing 1D finite element analysis of axial deformation from scratch, which can be instructive to compare with a FEniCS script producing the same vector of nodal displacements.  The intended use of the FEniCS demos is for an instructor to be able to show visualizations of deformations, stress fields, and strain fields, to augment the presentation of linear elasticity theory in the junior/senior-level course MAE 131B (Solid Mechanics II), as taught at UC San Diego following the textbook [*Intermediate Solid Mechanics*](https://doi.org/10.1017/9781108589000) by Marko and Vlado Lubarda.  

Python programming and FEniCS usage are not expected of students, but the code is likely still accessible to those who are especially interested.  Information on installing and running FEniCS can be found on the project's [homepage](https://fenicsproject.org/).  The scripts here currently use "legacy FEniCS", version 2019.2.0 (not the most recent version, FEniCSx, which remains under active development).  Finite element formulations based on energy minimization are favored, mainly to correspond with the textbook's discussion of finite element analysis in the context of energy methods.  
