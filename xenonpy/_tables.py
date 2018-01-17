# Copyright 2017 TsumiNa. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

import os

from sqlalchemy import Boolean, Column, Float, ForeignKey, Integer, String
from sqlalchemy.ext.associationproxy import association_proxy
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker

Base = declarative_base()


class Mendeleev(Base):
    '''
    Chemical element.

    Attributes:
      abundance_crust : float
        Abundance in the earth's crust in mg/kg
      abundance_sea : float
        Abundance in the seas in mg/L
      atomic_number : int
        Atomic number
      atomic_radius : float
        Atomic radius in pm
      atomic_radius_rahm : float
        Atomic radius by Rahm et al. in pm
      atomic_volume : float
        Atomic volume in cm3/mol
      atomic_weight : float
        Relative atomic weight as the ratio of the average mass of atoms
        of the element to 1/12 of the mass of an atom of 12C
      boiling_point : float
        Boiling temperature in K
      c6 : float
        C_6 dispersion coefficient in a.u. from X. Chu & A. Dalgarno, J. Chem. Phys.,
        121(9), 4083–4088 (2004) doi:10.1063/1.1779576, and the value for
        Hydrogen was taken from K. T. Tang, J. M. Norbeck and P. R. Certain,
        J. Chem. Phys. 64, 3063 (1976), doi:10.1063/1.432569
      c6_gb : float
        C_6 dispersion coefficient in a.u. from Gould, T., & Bučko, T. (2016).
        JCTC, 12(8), 3603–3613. http://doi.org/10.1021/acs.jctc.6b00361
      covalent_radius_bragg : float
        Covalent radius in pm from
      covalent_radius_cordero : float
        Covalent radius in pm from Cordero, B., Gómez, V., Platero-Prats, A.
        E., Revés, M., Echeverría, J., Cremades, E., … Alvarez, S. (2008).
        Covalent radii revisited. Dalton Transactions, (21), 2832.
        doi:10.1039/b801115j
      covalent_radius_pyykko : float
        Single bond covalent radius in pm Pyykkö, P., & Atsumi, M. (2009).
        Molecular Single-Bond Covalent Radii for Elements 1-118.
        Chemistry - A European Journal, 15(1), 186–197.
        doi:10.1002/chem.200800987
      covalent_radius_pyykko_double : float
        Double bond covalent radius in pm from P. Pyykkö et al.
      covalent_radius_pyykko_triple : float
        Triple bond covalent radius in pm from P. Pyykkö et al.
      covalent_radius_slater : float
        Covalent radius in pm from Slater
      density : float
        Density at 295K in g/cm3
      dipole_polarizability : float
        Dipole polarizability in atomic units from P. Schwerdtfeger "Table of
        experimental and calculated static dipole polarizabilities for the
        electronic ground states of the neutral elements (in atomic units)",
        February 11, 2014
      discovery_year: int
        The year the element was discovered
      electron_affinity : float
        Electron affinity in eV
      en_allen : float
        Allen's scale of electronegativity (Configurational energy)
      en_ghosh : float
        Ghosh's scale of enectronegativity
      en_pauling : float
        Pauling's scale of electronegativity
      econf : str
        Ground state electron configuration
      evaporation_heat : float
        Evaporation heat in kJ/mol
      fusion_heat : float
        Fusion heat in kJ/mol
      gas_basicity : Float
        Gas basicity
      group : int
        Group in periodic table
      heat_of_formation : float
        Heat of formation in kJ/mol
      is_monoisotopic : bool
        A flag marking if the element is monoisotopic
      lattice_constant : float
        Lattice constant in ang
      mass : float
        Relative atomic mass. Ratio of the average mass of atoms
        of the element to 1/12 of the mass of an atom of 12C
      melting_point : float
        Melting temperature in K
      metallic_radius : Float
        Single-bond metallic radius or metallic radius, have been
        calculated by Pauling using interatomic distances and an
        equation relating such distances with bond number
      metallic_radius_c12 : Float
        Metallic radius obtained by Pauling with an assumed number of
        nearest neighbors equal to 12
      period : int
        Period in periodic table
      proton_affinity : Float
        Proton affinity
      series : int
        Index to chemical series
      specific_heat : float
        Specific heat in J/g mol @ 20 C
      thermal_conductivity : float
        Thermal conductivity in @/m K @25 C
      vdw_radius : float
        Van der Waals radius in pm from W. M. Haynes, Handbook of Chemistry and
        Physics 95th Edition, CRC Press, New York, 2014, ISBN-10: 1482208679,
        ISBN-13: 978-1482208672.
      vdw_radius_bondi : float
        Van der Waals radius according to Bondi in pm
      vdw_radius_truhlar : float
        Van der Waals radius according to Truhlar in pm
      vdw_radius_rt : float
        Van der Waals radius according to Rowland and Taylor in pm
      vdw_radius_batsanov : float
        Van der Waals radius according to Batsanov in pm
      vdw_radius_dreiding : float
        Van der Waals radius from the DREIDING force field in pm
      vdw_radius_uff : float
        Van der Waals radius from the UFF in pm
      vdw_radius_mm3 : float
        Van der Waals radius from MM3 in pm
    '''

    __tablename__ = 'mendeleev'

    abundance_crust = Column(Float)
    abundance_sea = Column(Float)
    atomic_number = Column(Integer, primary_key=True)
    atomic_radius = Column(Float)
    atomic_radius_rahm = Column(Float)
    atomic_volume = Column(Float)
    atomic_weight = Column(Float)
    atomic_weight_uncertainty = Column(Float)
    boiling_point = Column(Float)
    covalent_radius_bragg = Column(Float)
    covalent_radius_cordero = Column(Float)
    covalent_radius_pyykko = Column(Float)
    covalent_radius_pyykko_double = Column(Float)
    covalent_radius_pyykko_triple = Column(Float)
    covalent_radius_slater = Column(Float)
    c6 = Column(Float)
    c6_gb = Column(Float)
    density = Column(Float)
    dipole_polarizability = Column(Float)
    electron_affinity = Column(Float)
    en_allen = Column(Float)
    en_ghosh = Column(Float)
    en_pauling = Column(Float)
    evaporation_heat = Column(Float)
    fusion_heat = Column(Float)
    gas_basicity = Column(Float)
    heat_of_formation = Column(Float)
    is_monoisotopic = Column(Boolean)
    is_radioactive = Column(Boolean)
    lattice_constant = Column(Float)
    melting_point = Column(Float)
    metallic_radius = Column(Float)
    metallic_radius_c12 = Column(Float)
    period = Column(Integer)
    proton_affinity = Column(Float)
    specific_heat = Column(Float)
    thermal_conductivity = Column(Float)
    vdw_radius = Column(Float)
    vdw_radius_alvarez = Column(Float)
    vdw_radius_bondi = Column(Float)
    vdw_radius_truhlar = Column(Float)
    vdw_radius_rt = Column(Float)
    vdw_radius_batsanov = Column(Float)
    vdw_radius_dreiding = Column(Float)
    vdw_radius_uff = Column(Float)
    vdw_radius_mm3 = Column(Float)