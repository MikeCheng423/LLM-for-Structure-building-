import pytest

from ase_auto_build.structure import (
    make_supercell,
    make_vacancy,
    per_atom_symbols,
    read_poscar,
    substitute,
    write_poscar,
)


def test_read_write_roundtrip(scf_case, tmp_path):
    struct = read_poscar(scf_case / "POSCAR")
    assert struct["elements"] == ["Al", "O"]
    assert struct["counts"] == [1, 1]
    assert struct["cartesian"] is False

    out = tmp_path / "POSCAR_OUT"
    write_poscar(struct, out)
    again = read_poscar(out)
    assert again["elements"] == struct["elements"]
    for a, b in zip(again["coords"], struct["coords"]):
        assert a == pytest.approx(b)


def test_make_supercell_direct(scf_case):
    struct = read_poscar(scf_case / "POSCAR")
    supercell = make_supercell(struct, (2, 1, 1))

    assert supercell["counts"] == [2, 2]
    assert supercell["lattice"][0][0] == pytest.approx(8.0)
    assert supercell["lattice"][1][1] == pytest.approx(4.0)
    # The original Al at x=0 replicates to x=0 and x=0.5 in the doubled cell.
    al_x = sorted(coord[0] for coord in supercell["coords"][:2])
    assert al_x == pytest.approx([0.0, 0.5])


def test_make_supercell_rejects_zero():
    with pytest.raises(ValueError):
        make_supercell({"lattice": [[1, 0, 0]] * 3, "coords": [], "flags": [], "counts": [], "cartesian": False}, (0, 1, 1))


def test_make_vacancy(scf_case):
    struct = read_poscar(scf_case / "POSCAR")
    vacant = make_vacancy(struct, 1)
    assert vacant["elements"] == ["O"]
    assert vacant["counts"] == [1]
    assert len(vacant["coords"]) == 1

    with pytest.raises(ValueError):
        make_vacancy(struct, 3)


def test_substitute(scf_case):
    struct = read_poscar(scf_case / "POSCAR")
    doped = substitute(struct, 2, "Mg")
    assert per_atom_symbols(doped) == ["Al", "Mg"]
    assert doped["elements"] == ["Al", "Mg"]
    assert doped["coords"] == struct["coords"]


# --------------------------------------------------- builder operations

from ase_auto_build.structure import (  # noqa: E402
    build_struct,
    cell_from_parameters,
    cell_parameters,
    combine_structures,
    coordination,
    delete_atoms,
    set_cell,
    wrap_to_cell,
)


def _rocksalt():
    symbols, coords = [], []
    for symbol, base in (
        ("Na", [[0, 0, 0], [.5, .5, 0], [.5, 0, .5], [0, .5, .5]]),
        ("Cl", [[.5, 0, 0], [0, .5, 0], [0, 0, .5], [.5, .5, .5]]),
    ):
        for b in base:
            symbols.append(symbol)
            coords.append(b)
    return build_struct("NaCl", [[5.64, 0, 0], [0, 5.64, 0], [0, 0, 5.64]], symbols, coords)


def test_cell_parameters_roundtrip():
    lattice = cell_from_parameters(4.1, 5.2, 6.3, 80.0, 95.0, 112.0)
    params = cell_parameters(lattice)
    assert params["a"] == pytest.approx(4.1)
    assert params["b"] == pytest.approx(5.2)
    assert params["c"] == pytest.approx(6.3)
    assert params["alpha"] == pytest.approx(80.0)
    assert params["beta"] == pytest.approx(95.0)
    assert params["gamma"] == pytest.approx(112.0)


def test_cell_from_parameters_rejects_impossible():
    with pytest.raises(ValueError):
        cell_from_parameters(1, 1, 1, 10, 170, 90)
    with pytest.raises(ValueError):
        cell_from_parameters(-1, 1, 1, 90, 90, 90)


def test_set_cell_preserve_fractional(scf_case):
    struct = read_poscar(scf_case / "POSCAR")
    bigger = set_cell(struct, [[8, 0, 0], [0, 8, 0], [0, 0, 8]], preserve="fractional")
    assert bigger["coords"][1] == pytest.approx([0.5, 0.5, 0.5])
    assert bigger["lattice"][0][0] == pytest.approx(8.0)


def test_set_cell_preserve_cartesian(scf_case):
    struct = read_poscar(scf_case / "POSCAR")
    bigger = set_cell(struct, [[8, 0, 0], [0, 8, 0], [0, 0, 8]], preserve="cartesian")
    # Atom was at 0.5 frac of a 4 Å cell = 2 Å = 0.25 frac of the 8 Å cell.
    assert bigger["coords"][1] == pytest.approx([0.25, 0.25, 0.25])


def test_wrap_to_cell():
    struct = build_struct("t", [[3, 0, 0], [0, 3, 0], [0, 0, 3]], ["H"], [[1.2, -0.3, 0.5]])
    wrapped = wrap_to_cell(struct)
    assert wrapped["coords"][0] == pytest.approx([0.2, 0.7, 0.5])


def test_delete_atoms():
    struct = _rocksalt()
    fewer = delete_atoms(struct, [1, 2, 5])
    assert fewer["elements"] == ["Na", "Cl"]
    assert fewer["counts"] == [2, 3]
    with pytest.raises(ValueError):
        delete_atoms(struct, [])


def test_rotate_atoms():
    from ase_auto_build.structure import rotate_atoms
    cube = [[10, 0, 0], [0, 10, 0], [0, 0, 10]]
    struct = build_struct("t", cube, ["H", "H", "He"],
                          [[0.5, 0.5, 0.5], [0.7, 0.5, 0.5], [0.1, 0.1, 0.1]])
    # 90° about z, explicit center on atom 1: atom 2 swings from +x to +y.
    turned = rotate_atoms(struct, [2], "z", 90.0, center=(5, 5, 5))
    assert turned["coords"][1] == pytest.approx([0.5, 0.7, 0.5])
    # 180° about the centroid of atoms 1+2 swaps them; atom 3 is untouched.
    swapped = rotate_atoms(struct, [1, 2], (0, 0, 1), 180.0)
    assert swapped["coords"][0] == pytest.approx([0.7, 0.5, 0.5])
    assert swapped["coords"][1] == pytest.approx([0.5, 0.5, 0.5])
    assert swapped["coords"][2] == pytest.approx([0.1, 0.1, 0.1])
    with pytest.raises(ValueError):
        rotate_atoms(struct, [1], "q", 90.0)
    with pytest.raises(ValueError):
        rotate_atoms(struct, [1], (0, 0, 0), 90.0)


def test_coordination_rocksalt_and_h2():
    nacl = _rocksalt()
    assert coordination(nacl)[0]["coordination"] == 6
    h2 = build_struct("H2", [[10, 0, 0], [0, 10, 0], [0, 0, 10]],
                      ["H", "H"], [[.5, .5, .463], [.5, .5, .537]])
    assert [a["coordination"] for a in coordination(h2)] == [1, 1]
    # fcc primitive cell: 12 nearest neighbours, all through periodic images.
    al = build_struct("Al", [[0, 2.025, 2.025], [2.025, 0, 2.025], [2.025, 2.025, 0]],
                      ["Al"], [[0, 0, 0]])
    assert coordination(al)[0]["coordination"] == 12


def test_combine_stack_au_on_sheet():
    host = build_struct("sheet", [[4.92, 0, 0], [-2.46, 4.26, 0], [0, 0, 8]],
                        ["C"] * 4, [[0, 0, .25], [1/3, 2/3, .25], [.5, .5, .25], [5/6, 1/6, .25]])
    guest = build_struct("Au", [[2.88, 0, 0], [0, 2.88, 0], [0, 0, 2.88]], ["Au"], [[0, 0, 0]])
    combo = combine_structures(host, guest, mode="stack", gap=2.5, vacuum=12.0)

    assert combo["elements"] == ["C", "Au"]
    assert combo["counts"] == [4, 1]
    # In-plane host vectors unchanged.
    assert combo["lattice"][0] == pytest.approx([4.92, 0, 0])
    z_of = [c[2] * combo["lattice"][2][2] for c in combo["coords"]]
    # Au sits gap above the carbon sheet (sheet at z = 0.25*8 = 2 Å).
    assert z_of[4] - max(z_of[:4]) == pytest.approx(2.5)
    # And vacuum above the Au.
    assert combo["lattice"][2][2] - z_of[4] == pytest.approx(12.0)


def test_combine_insert_keeps_host_cell():
    host = _rocksalt()
    guest = build_struct("H", [[2, 0, 0], [0, 2, 0], [0, 0, 2]], ["H"], [[.5, .5, .5]])
    combo = combine_structures(host, guest, mode="insert", gap=0.0, shift=(1.0, 1.0))
    assert combo["lattice"] == host["lattice"]
    assert combo["counts"] == [4, 4, 1]


def test_combine_preserves_selective_flags():
    host = build_struct("slab", [[4, 0, 0], [0, 4, 0], [0, 0, 8]],
                        ["Al", "Al"], [[0, 0, .1], [.5, .5, .2]],
                        flags=[["F", "F", "F"], ["T", "T", "T"]])
    guest = build_struct("H", [[2, 0, 0], [0, 2, 0], [0, 0, 2]], ["H"], [[0, 0, 0]])
    combo = combine_structures(host, guest, mode="stack", gap=2.0, vacuum=5.0)
    assert combo["selective"] is True
    assert combo["flags"][0] == ["F", "F", "F"]
    assert combo["flags"][2] == ["T", "T", "T"]


def test_combine_rejects_bad_mode():
    host = _rocksalt()
    with pytest.raises(ValueError):
        combine_structures(host, host, mode="sideways")
