=======
Dataset
=======

XenonPy embed with ``elements`` and ``elements_completed`` dataset for descriptors calculation.
These data are summarized from `mendeleev`_, `pymatgen`_, `CRC Hand Book`_ and `Magpie`_.

``elements`` dataset contains 118 elements with 74 elemental features(incomplete).
``elements_completed`` is imputed version by use multiple imputation chained equation [1]_.

Because a large number of missing data in high atomic number elements lead to difficulty impute.
The ``elements_completed`` dataset only have 94 elements from `H` to `Pu` with 58 features.

.. _CRC Hand Book: http://hbcponline.com/faces/contents/ContentsSearch.xhtml
.. _Magpie: https://bitbucket.org/wolverton/magpie
.. _mendeleev: https://mendeleev.readthedocs.io
.. _pymatgen: http://pymatgen.org/

.. [1] Int J Methods Psychiatr Res. 2011 Mar 1; 20(1): 40–49.
            doi: `10.1002/mpr.329 <10.1002/mpr.329>`_

The followig data are currently available in ``elements``:

=================================   ===================================================================================
    feature                             description
---------------------------------   -----------------------------------------------------------------------------------
``period``                          Period in the periodic table.
``atomic_number``                   Number of protons found in the nucleus of an atom.
``mendeleev_number``                Atom number in mendeleev's periodic table
``atomic_radius``                   Atomic radius.
``atomic_radius_rahm``              Atomic radius by Rahm et al.
``atomic_volume``                   Atomic volume.
``atomic_weight``                   The mass of an atom.
``icsd_volume``                     Atom volume in ICSD database.
``lattice_constant``                Physical dimension of unit cells in a crystal lattice.
``vdw_radius``                      Van der Waals radiusis.
``vdw_radius_alvarez``              Van der Waals radius according to Alvarez.
``vdw_radius_batsanov``             Van der Waals radius according to Batsanov.
``vdw_radius_bondi``                Van der Waals radius according to Bondi.
``vdw_radius_dreiding``             Van der Waals radius from the DREIDING FF.
``vdw_radius_mm3``                  Van der Waals radius from the MM3 FF.
``vdw_radius_rt``                   Van der Waals radius according to Rowland and Taylor.
``vdw_radius_truhlar``              Van der Waals radius according to Truhlar.
``vdw_radius_uff``                  Van der Waals radius from the UFF.
``covalent_radius_bragg``           Covalent radius by Bragg
``covalent_radius_cordero``         Covalent radius by Cerdero et al.
``covalent_radius_pyykko``          Single bond covalent radius by Pyykko et al.
``covalent_radius_pyykko_double``   Double bond covalent radius by Pyykko et al.
``covalent_radius_pyykko_triple``   Triple bond covalent radius by Pyykko et al.
``covalent_radius_slater``          Covalent radius by Slater.
``c6``                              C_6 dispersion coefficient in a.u.
``c6_gb``                           C_6 dispersion coefficient in a.u.
``density``                         Density at 295K.
``proton_affinity``                 Proton affinity.
``dipole_polarizability``           Dipole polarizability.
``electron_affinity``               Electron affinity.
``electron_negativity``             Tendency of an atom to attract a shared pair of electronsElectron affinity.
``en_allen``                        Allen's scale of electronegativity.
``en_ghosh``                        Ghosh's scale of electronegativity.
``en_pauling``                      Mulliken's scale of electronegativity.
``gs_bandgap``                      DFT bandgap energy of T=0K ground state.
``gs_energy``                       Estimated FCC lattice parameter based on the DFT volume.
``gs_est_bcc_latcnt``               Estimated BCC lattice parameter based on the DFT volume.
``gs_est_fcc_latcnt``               Estimated FCC lattice parameter based on the DFT volume.
``gs_mag_moment``                   DFT magnetic momenet of T=0K ground state.
``gs_volume_per``                   DFT volume per atom of T=0K ground state.
``hhi_p``                           Herfindahl−Hirschman Index (HHI) production values
``hhi_r``                           Herfindahl−Hirschman Index (HHI) reserves values
``specific_heat``                   Specific heat at 20 C.
``gas_basicity``                    Gas basicity.
``first_ion_en``                    First ionisation energy.
``fusion_enthalpy``                 Fusion heat.
``heat_of_formation``               Heat of formation.
``heat_capacity_mass``              Mass specific heat capacity.
``heat_capacity_molar``             Molar specific heat capacity.
``evaporation_heat``                Evaporation heat.
``linear_expansion_coefficient``    Coefficient of linear expansion.
``boiling_point``                   Boiling temperature.
``brinell_hardness``                Brinell Hardness Number.
``bulk_modulus``                    Bulk modulus.
``melting_point``                   Melting point.
``metallic_radius``                 Single-bond metallic radius.
``metallic_radius_c12``             Metallic radius with 12 nearest neighbors
``thermal_conductivity``            Thermal conductivity at 25 C.
``sound_velocity``                  Speed of sound.
``vickers_hardness``                Value of Vickers hardness test.
``Polarizability``                  Ability to form instantaneous dipoles.
``youngs_modulus``                  Young's modulus.
``poissons_ratio``                  Poisson's ratio.
``molar_volume``                    Molar volume.
``num_unfilled``                    Total unfilled electron.
``num_valance``                     Total valance electron.
``num_d_unfilled``                  Unfilled electron in d shell.
``num_d_valence``                   Valance electron in d shell.
``num_f_unfilled``                  Unfilled electron in f shell.
``num_f_valence``                   Valance electron in d shell.
``num_p_unfilled``                  Unfilled electron in p shell.
``num_p_valence``                   Valance electron in d shell.
``num_s_unfilled``                  Unfilled electron in s shell.
``num_s_valence``                   Valance electron in d shell.
=================================   ===================================================================================


Load dataset
============

You can use :class:`~xenonpy.utils.Loader` to load preset dataset.
See `loader_saver sample <https://github.com/yoshida-lab/XenonPy/blob/master/samples/load_and_save_data.ipynb>`_ for details.
