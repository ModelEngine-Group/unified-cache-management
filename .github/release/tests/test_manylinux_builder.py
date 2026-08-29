from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def test_wheel_builder_repairs_to_the_planned_manylinux_tag() -> None:
    dockerfile = (
        ROOT / ".github" / "release" / "docker" / "Dockerfile.wheel"
    ).read_text(encoding="utf-8")

    assert "auditwheel==6.7.0" in (
        ROOT / ".github" / "release" / "requirements" / "wheel-build.txt"
    ).read_text(encoding="utf-8")
    assert "--outdir /tmp/ucm-raw-wheel" in dockerfile
    assert "mapfile -t auditwheel_excludes" in dockerfile
    assert 'repair_args+=(--exclude "${library}")' in dockerfile
    assert '--plat "${UCM_TARGET_PLATFORM_TAG}"' in dockerfile
    assert "--only-plat" in dockerfile
    assert "ucm-python -m auditwheel -v repair" in dockerfile
    assert 'SOURCE_DATE_EPOCH="${SOURCE_DATE_EPOCH}"' in dockerfile
    assert "--strip" not in dockerfile
    assert "tee /out/auditwheel-repair.txt" in dockerfile
    assert "grep -Fq 'Grafting:' /out/auditwheel-repair.txt" in dockerfile
    assert "auditwheel grafted an unplanned external library" in dockerfile
    assert 'auditwheel -v show "${repaired_wheel}"' in dockerfile
    assert "tee /out/auditwheel-show.txt" in dockerfile


def test_wheel_workflow_materializes_external_roots_for_the_builder() -> None:
    workflow = (ROOT / ".github" / "workflows" / "_build-wheel.yml").read_text(
        encoding="utf-8"
    )

    assert ".external_runtime_exclude_patterns[]" in workflow
    assert ">.ucm-compact/auditwheel-exclude-patterns.txt" in workflow
    assert 'index("\\n") == null and index("\\r") == null' in workflow
    assert "source_date_epoch=$(git show" in workflow.replace('"', "")
    assert (
        'build-arg "SOURCE_DATE_EPOCH=${{ steps.task.outputs.source_date_epoch }}"'
        in workflow
    )
    assert "UCM_TARGET_PLATFORM_TAG=$(jq -r '.target_platform_tag'" in workflow
    assert "auditwheel==6.4.2" not in workflow
    assert "python -m auditwheel -v show" not in workflow
    assert "compact record-wheel-result" in workflow
    assert "test -s out/wheel/auditwheel-repair.txt" in workflow
    assert "test -s out/wheel/auditwheel-show.txt" in workflow


def test_ucm_owned_metrics_library_has_wheel_relative_install_rpaths() -> None:
    expected = {
        "ucm/shared/metrics/CMakeLists.txt": 'INSTALL_RPATH "$ORIGIN"',
        "ucm/store/cache/CMakeLists.txt": (
            'INSTALL_RPATH "$ORIGIN/../../shared/metrics"'
        ),
        "ucm/store/posix/CMakeLists.txt": (
            'INSTALL_RPATH "$ORIGIN/../../shared/metrics"'
        ),
        "ucm/store/pipeline/CMakeLists.txt": (
            'INSTALL_RPATH "$ORIGIN/../../shared/metrics"'
        ),
    }
    for relative_path, contract in expected.items():
        assert contract in (ROOT / relative_path).read_text(encoding="utf-8")


def test_cuda_nfsstore_propagates_its_runtime_dependency() -> None:
    cmake = (
        ROOT / "ucm" / "store" / "nfsstore" / "device" / "cuda" / "CMakeLists.txt"
    ).read_text(encoding="utf-8")

    assert "target_link_directories(storedevice PUBLIC ${CUDA_ROOT}/lib64)" in cmake
    assert (
        "target_link_libraries(storedevice PUBLIC infra_status infra_logger cudart)"
        in cmake
    )
