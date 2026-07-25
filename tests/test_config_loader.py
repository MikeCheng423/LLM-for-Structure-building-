from vasp_auto.config_loader import load_config, merge_local_config


def test_load_config_explicit_path(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        "vasp_executable: vasp_std\n"
        "jobs_root: jobs\n"
        "potcar_root: POTCAR\n"
        "potcar_map:\n  Fe: Fe_pv\n",
        encoding="utf-8",
    )

    config = load_config(str(config_file))

    assert config["_config_path"] == str(config_file)
    # potcar_root resolves relative to the config file location.
    assert config["potcar_root"] == str(tmp_path / "POTCAR")
    assert config["potcar_map"] == {"Fe": "Fe_pv"}


def test_load_config_jobs_root_follows_cwd(tmp_path, monkeypatch):
    config_file = tmp_path / "config.yaml"
    config_file.write_text("jobs_root: jobs\n", encoding="utf-8")
    workdir = tmp_path / "elsewhere"
    workdir.mkdir()
    monkeypatch.chdir(workdir)

    config = load_config(str(config_file))

    assert config["jobs_root"] == str(workdir / "jobs")


def test_merge_local_config_overrides_per_case(tmp_path):
    global_file = tmp_path / "config.yaml"
    global_file.write_text("vasp_executable: vasp_std\nneb_images: 5\n", encoding="utf-8")
    config = load_config(str(global_file))

    case_dir = tmp_path / "case1"
    case_dir.mkdir()
    (case_dir / "config.yaml").write_text(
        "neb_images: 9\npotcar_root: LOCAL_POTCAR\n", encoding="utf-8"
    )

    merged = merge_local_config(config, case_dir)

    assert merged["neb_images"] == 9
    assert merged["potcar_root"] == str(case_dir / "LOCAL_POTCAR")
    assert merged["vasp_executable"] == config["vasp_executable"]
    # The base config object is untouched.
    assert config["neb_images"] == 5


def test_merge_local_config_without_local_file(tmp_path):
    global_file = tmp_path / "config.yaml"
    global_file.write_text("neb_images: 5\n", encoding="utf-8")
    config = load_config(str(global_file))

    empty = tmp_path / "nothing"
    empty.mkdir()
    assert merge_local_config(config, empty) is config


def test_load_config_finds_cwd_config(tmp_path, monkeypatch):
    (tmp_path / "config.yaml").write_text("neb_images: 7\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    config = load_config()

    assert config["neb_images"] == 7
    assert config["_config_dir"] == str(tmp_path)
