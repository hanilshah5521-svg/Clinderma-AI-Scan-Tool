"""
CLINDERMA STAGE 3C — ROBUST FRESH DATASET ACQUISITION

Fixes based on the actual Stage 3B run:
1. MEMI-DS Figshare article contains individual files, not one archive.
   We download every attached file individually.
2. Roboflow SDK versions differ. We therefore use the Roboflow REST API
   for authenticated discovery, then use the SDK only for the selected
   project/version download.
3. Nothing is normalized, split, trained, or deleted.

Large data is stored in:
    /content/clinderma_workspace/raw/

Persistent Drive is only used for the project code.
"""

from pathlib import Path
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
import urllib.error

WORKSPACE = Path("/content/clinderma_workspace")
RAW = WORKSPACE / "raw"
MANIFEST = WORKSPACE / "dataset_manifest.json"

MEMI_ARTICLE_ID = "29209229"
MEMI_API = f"https://api.figshare.com/v2/articles/{MEMI_ARTICLE_ID}"

ROBOFLOW_API_BASE = "https://api.roboflow.com"


# ================================================================
# GENERAL UTILITIES
# ================================================================

def get_key():
    """Read ROBOFLOW_API_KEY from Colab Secrets, then environment."""
    try:
        from google.colab import userdata
        key = userdata.get("ROBOFLOW_API_KEY")
        if key:
            return key
    except Exception:
        pass

    return os.environ.get("ROBOFLOW_API_KEY")


def safe_name(value):
    value = str(value)
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def list_files(path):
    if not path.exists():
        return []
    return [p for p in path.rglob("*") if p.is_file()]


def save_manifest(manifest):
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(MANIFEST) + ".tmp")
    tmp.write_text(
        json.dumps(manifest, indent=2, default=str),
        encoding="utf-8",
    )
    tmp.replace(MANIFEST)


def http_get_json(url, headers=None):
    request = urllib.request.Request(
        url,
        headers=headers or {
            "User-Agent": "Clinderma-Pigmentation/1.0",
            "Accept": "application/json",
        },
    )

    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            status = response.status
            body = response.read()
            content_type = response.headers.get("Content-Type", "")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"HTTP {exc.code} while requesting {url}: {body[:1000]}"
        ) from exc

    if status != 200:
        raise RuntimeError(
            f"HTTP {status} while requesting {url}"
        )

    try:
        return json.loads(body.decode("utf-8"))
    except Exception as exc:
        raise RuntimeError(
            f"Expected JSON from {url}, got Content-Type={content_type}"
        ) from exc


def download_file(url, destination, headers=None):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    request = urllib.request.Request(
        url,
        headers=headers or {
            "User-Agent": "Clinderma-Pigmentation/1.0",
        },
    )

    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            status = response.status
            content_type = response.headers.get("Content-Type", "")

            if status != 200:
                raise RuntimeError(
                    f"HTTP {status} while downloading {url}"
                )

            with open(destination, "wb") as f:
                shutil.copyfileobj(response, f)

    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"HTTP {exc.code} downloading {url}: {body[:1000]}"
        ) from exc

    size = destination.stat().st_size

    if size <= 0:
        raise RuntimeError(
            f"Downloaded zero-byte file: {destination}"
        )

    return {
        "size_bytes": size,
        "content_type": content_type,
    }


# ================================================================
# MEMI-DS — FIGSHARE INDIVIDUAL FILE DOWNLOAD
# ================================================================

def download_memi_ds():
    target = RAW / "memi_ds"

    existing = list_files(target)

    # A valid prior download is reused rather than downloaded twice.
    if len(existing) > 1:
        print("\nMEMI-DS already exists in workspace.")
        print("Files:", len(existing))
        return {
            "status": "EXISTING",
            "source": "figshare",
            "article_id": MEMI_ARTICLE_ID,
            "path": str(target),
            "files": len(existing),
        }

    target.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("MEMI-DS — FIGSHARE INDIVIDUAL FILE DOWNLOAD")
    print("=" * 80)

    article = http_get_json(MEMI_API)

    title = article.get("title", "")
    attached = article.get("files", [])

    print("Article:", MEMI_ARTICLE_ID)
    print("Title:", title)
    print("Attached files:", len(attached))

    if not attached:
        raise RuntimeError(
            "Figshare article returned zero attached files."
        )

    downloaded = []
    failed = []

    for index, file_info in enumerate(attached, start=1):
        name = file_info.get("name")
        url = file_info.get("download_url")

        if not name or not url:
            failed.append(
                {
                    "index": index,
                    "name": name,
                    "error": "Missing filename or download URL",
                }
            )
            continue

        # Figshare MEMI files are individual JPG/PNG/JSON files.
        # Preserve the filename; do not try to extract it.
        local_name = safe_name(name)
        destination = target / local_name

        if destination.exists() and destination.stat().st_size > 0:
            downloaded.append(
                {
                    "name": name,
                    "url": url,
                    "status": "EXISTING",
                    "size_bytes": destination.stat().st_size,
                }
            )
            continue

        try:
            info = download_file(url, destination)

            downloaded.append(
                {
                    "name": name,
                    "url": url,
                    "status": "DOWNLOADED",
                    **info,
                }
            )

            if index == 1 or index % 25 == 0 or index == len(attached):
                print(
                    f"MEMI progress: {index}/{len(attached)} files"
                )

        except Exception as exc:
            failed.append(
                {
                    "index": index,
                    "name": name,
                    "url": url,
                    "error": repr(exc),
                }
            )

    # Keep a source record.
    (target / "SOURCE.txt").write_text(
        "Dataset: MEMI-DS\n"
        f"Figshare article: {MEMI_ARTICLE_ID}\n"
        f"Retrieved: {time.strftime('%Y-%m-%d %H:%M:%S')}\n",
        encoding="utf-8",
    )

    actual_files = [
        p for p in target.rglob("*")
        if p.is_file() and p.name != "SOURCE.txt"
    ]

    result = {
        "status": (
            "DOWNLOADED"
            if not failed
            else "PARTIAL"
        ),
        "source": "figshare",
        "article_id": MEMI_ARTICLE_ID,
        "title": title,
        "path": str(target),
        "files": len(actual_files),
        "expected_files": len(attached),
        "downloaded_records": downloaded,
        "failed_records": failed,
    }

    print("\nMEMI FINAL:")
    print("Expected:", len(attached))
    print("Present:", len(actual_files))
    print("Failed:", len(failed))

    if failed:
        print(
            "\nWARNING: Some MEMI files failed. "
            "We will NOT proceed to normalization until this is resolved."
        )

    return result


# ================================================================
# ROBOFLOW REST API DISCOVERY
# ================================================================

def roboflow_request(path, key):
    """
    Query Roboflow REST API using the API key as a query parameter.
    """
    separator = "&" if "?" in path else "?"
    url = f"{ROBOFLOW_API_BASE}{path}{separator}api_key={key}"

    return http_get_json(
        url,
        headers={
            "User-Agent": "Clinderma-Pigmentation/1.0",
            "Accept": "application/json",
        },
    )


def discover_roboflow():
    key = get_key()

    if not key:
        raise RuntimeError(
            "ROBOFLOW_API_KEY was not found in Colab Secrets."
        )

    print("\n" + "=" * 80)
    print("ROBOFLOW — REST API ACCOUNT DISCOVERY")
    print("=" * 80)

    # Root endpoint is useful because Roboflow's own error messages
    # recommend it for discovering the active workspace.
    root = roboflow_request("/", key)

    print("\nAuthenticated Roboflow response:")
    print(json.dumps(root, indent=2)[:5000])

    rows = []

    # Different API responses expose workspace information slightly
    # differently, so normalize several common structures.
    workspaces = []

    if isinstance(root, dict):
        if isinstance(root.get("workspaces"), list):
            workspaces.extend(root["workspaces"])

        if isinstance(root.get("workspace"), dict):
            workspaces.append(root["workspace"])

        if isinstance(root.get("workspace"), str):
            workspaces.append({"id": root["workspace"]})

        # Some responses put the active workspace under "name".
        if root.get("name") and (
            root.get("id")
            or root.get("workspace")
        ):
            workspaces.append(root)

    # Remove duplicate workspace identifiers.
    seen_ws = set()

    normalized_ws = []

    for ws in workspaces:
        if isinstance(ws, str):
            wid = ws
            wname = ws
        else:
            wid = (
                ws.get("id")
                or ws.get("workspace")
                or ws.get("workspace_id")
            )
            wname = (
                ws.get("name")
                or ws.get("workspace_name")
                or wid
            )

        if not wid or wid in seen_ws:
            continue

        seen_ws.add(wid)

        normalized_ws.append(
            {
                "id": wid,
                "name": wname,
            }
        )

    # If root did not provide enough information, try the documented
    # /workspaces endpoint.
    if not normalized_ws:
        try:
            data = roboflow_request("/workspaces", key)

            candidates = []
            if isinstance(data, dict):
                candidates = (
                    data.get("workspaces")
                    or data.get("items")
                    or []
                )
            elif isinstance(data, list):
                candidates = data

            for ws in candidates:
                if isinstance(ws, dict):
                    wid = (
                        ws.get("id")
                        or ws.get("workspace")
                    )
                    wname = (
                        ws.get("name")
                        or ws.get("workspace_name")
                        or wid
                    )
                else:
                    wid = str(ws)
                    wname = str(ws)

                if wid and wid not in seen_ws:
                    seen_ws.add(wid)
                    normalized_ws.append(
                        {
                            "id": wid,
                            "name": wname,
                        }
                    )
        except Exception as exc:
            print(
                "\n/workspaces endpoint was unavailable:",
                repr(exc),
            )

    if not normalized_ws:
        raise RuntimeError(
            "\nRoboflow authentication succeeded enough to query the API, "
            "but no workspace could be resolved.\n"
            "The printed authenticated response above is required to "
            "identify the correct workspace."
        )

    print("\nResolved workspaces:")
    for ws in normalized_ws:
        print(
            f"  {ws['name']} | id={ws['id']}"
        )

    # Query projects for each resolved workspace.
    for ws in normalized_ws:
        wid = ws["id"]

        possible_paths = [
            f"/{wid}",
            f"/{wid}/projects",
        ]

        found_projects = []

        for path in possible_paths:
            try:
                data = roboflow_request(path, key)

                if isinstance(data, dict):
                    candidates = (
                        data.get("projects")
                        or data.get("items")
                        or []
                    )

                    # A single project may also be returned.
                    if data.get("id") and data.get("name"):
                        candidates.append(data)

                elif isinstance(data, list):
                    candidates = data
                else:
                    candidates = []

                for project in candidates:
                    if isinstance(project, dict):
                        pid = (
                            project.get("id")
                            or project.get("project_id")
                        )
                        pname = (
                            project.get("name")
                            or project.get("project_name")
                            or pid
                        )
                        ptype = project.get("type")
                    else:
                        continue

                    if pid:
                        found_projects.append(
                            {
                                "workspace_id": wid,
                                "workspace_name": ws["name"],
                                "project_id": pid,
                                "project_name": pname,
                                "type": ptype,
                            }
                        )

            except Exception:
                # Try the next known endpoint shape.
                continue

        # Deduplicate.
        seen = set()

        for row in found_projects:
            key_tuple = (
                row["workspace_id"],
                row["project_id"],
            )

            if key_tuple in seen:
                continue

            seen.add(key_tuple)
            rows.append(row)

    # Deduplicate all workspaces/projects.
    final = []
    seen = set()

    for row in rows:
        key_tuple = (
            row["workspace_id"],
            row["project_id"],
        )

        if key_tuple not in seen:
            seen.add(key_tuple)
            final.append(row)

    if not final:
        raise RuntimeError(
            "\nNo accessible Roboflow projects were discovered.\n"
            "The authenticated response above is intentionally printed "
            "so we can diagnose the account/workspace permissions."
        )

    print("\n" + "=" * 80)
    print("ACCESSIBLE ROBOFLOW PROJECTS")
    print("=" * 80)

    for i, row in enumerate(final):
        print(
            f"[{i}] "
            f"{row['workspace_name']} / "
            f"{row['project_name']} "
            f"| project_id={row['project_id']} "
            f"| type={row['type']}"
        )

    return final


# ================================================================
# ROBOFLOW SDK DOWNLOAD
# ================================================================

def ensure_roboflow_sdk():
    try:
        import roboflow
    except ImportError:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-q",
                "roboflow",
            ],
            check=True,
        )


def get_version_id(version):
    if isinstance(version, dict):
        return (
            version.get("version")
            or version.get("id")
            or version.get("version_id")
        )

    return (
        getattr(version, "version", None)
        or getattr(version, "id", None)
        or getattr(version, "version_id", None)
    )


def download_roboflow_project(row):
    ensure_roboflow_sdk()

    from roboflow import Roboflow

    key = get_key()
    rf = Roboflow(api_key=key)

    workspace = rf.workspace(row["workspace_id"])
    project = workspace.project(row["project_id"])

    versions = project.versions()

    if isinstance(versions, dict):
        versions = (
            versions.get("versions")
            or versions.get("items")
            or []
        )

    if not isinstance(versions, list):
        versions = getattr(
            versions,
            "versions",
            [],
        ) or []

    if not versions:
        raise RuntimeError(
            f"No versions available for {row['project_name']}"
        )

    print(
        f"\nVersions for {row['project_name']}:"
    )

    for i, v in enumerate(versions):
        print(
            f"  [{i}] {get_version_id(v)}"
        )

    raw = input(
        "Choose version index "
        "(Enter = first/latest-listed): "
    ).strip()

    if raw == "":
        version = versions[0]
    else:
        idx = int(raw)
        version = versions[idx]

    vid = get_version_id(version)

    target = (
        RAW
        / f"{safe_name(row['workspace_name'])}"
        / f"{safe_name(row['project_name'])}"
        / f"v{safe_name(vid)}"
    )

    target.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("\nDownloading Roboflow dataset...")
    print("Destination:", target)

    version.download(
        "coco-segmentation",
        location=str(target),
    )

    count = len(list_files(target))

    if count == 0:
        raise RuntimeError(
            "Roboflow download returned but no files were found."
        )

    return {
        "status": "DOWNLOADED",
        "source": "roboflow",
        "workspace_id": row["workspace_id"],
        "workspace_name": row["workspace_name"],
        "project_id": row["project_id"],
        "project_name": row["project_name"],
        "version": vid,
        "format": "coco-segmentation",
        "path": str(target),
        "files": count,
    }


# ================================================================
# MAIN
# ================================================================

def main():
    WORKSPACE.mkdir(
        parents=True,
        exist_ok=True,
    )

    RAW.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 80)
    print("CLINDERMA STAGE 3C — FRESH DATASET ACQUISITION")
    print("=" * 80)

    key = get_key()

    print("\nROBOFLOW API KEY FOUND:", bool(key))

    if not key:
        raise RuntimeError(
            "ROBOFLOW_API_KEY is missing from Colab Secrets."
        )

    manifest = {
        "generated_at": time.strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
        "workspace": str(WORKSPACE),
        "datasets": {},
    }

    # ------------------------------------------------------------
    # MEMI
    # ------------------------------------------------------------

    try:
        manifest["datasets"]["memi_ds"] = (
            download_memi_ds()
        )
    except Exception as exc:
        manifest["datasets"]["memi_ds"] = {
            "status": "FAILED",
            "error": repr(exc),
        }

        print("\nMEMI-DS FAILED:")
        print(repr(exc))

    # ------------------------------------------------------------
    # ROBOFLOW
    # ------------------------------------------------------------

    projects = discover_roboflow()

    print("\n" + "=" * 80)
    print("SELECT ROBOFLOW PROJECTS")
    print("=" * 80)
    print(
        "Enter comma-separated project numbers.\n"
        "Example: 0,2,5\n"
        "Enter 'none' to skip."
    )

    selection = input("Selection: ").strip().lower()

    if selection != "none":
        indices = [
            int(x.strip())
            for x in selection.split(",")
            if x.strip()
        ]

        for idx in indices:
            if idx < 0 or idx >= len(projects):
                raise ValueError(
                    f"Invalid project index: {idx}"
                )

            row = projects[idx]

            key_name = (
                "roboflow__"
                f"{safe_name(row['workspace_name'])}__"
                f"{safe_name(row['project_name'])}"
            )

            try:
                result = download_roboflow_project(
                    row
                )
            except Exception as exc:
                result = {
                    "status": "FAILED",
                    "workspace_id": row["workspace_id"],
                    "workspace_name": row["workspace_name"],
                    "project_id": row["project_id"],
                    "project_name": row["project_name"],
                    "error": repr(exc),
                }

                print(
                    "\nROBOFLOW DOWNLOAD FAILED:",
                    repr(exc),
                )

            manifest["datasets"][key_name] = result

    save_manifest(manifest)

    print("\n" + "=" * 80)
    print("STAGE 3C SUMMARY")
    print("=" * 80)

    for name, result in manifest["datasets"].items():
        print("\n" + "-" * 80)
        print(name)
        print("Status:", result.get("status"))
        print("Path:", result.get("path", "N/A"))

        if result.get("files") is not None:
            print("Files:", result["files"])

        if result.get("expected_files") is not None:
            print(
                "Expected files:",
                result["expected_files"],
            )

        if result.get("error"):
            print("Error:", result["error"])

    print("\nManifest:")
    print(MANIFEST)

    print("\n" + "=" * 80)
    print("NO NORMALIZATION")
    print("NO SPLITS")
    print("NO TRAINING")
    print("NO DRIVE DATA DELETED")
    print("=" * 80)


if __name__ == "__main__":
    main()
